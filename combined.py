import cv2
import numpy as np
import os
import json
import base64
import logging
import schedule
import time
import requests
from datetime import datetime
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

# ---------------------- CONFIG AND PATHS ----------------------
# Set up directory paths
import_path = os.path.dirname(os.path.abspath(__file__))
ALERT_DATA_FOLDER = os.path.join(import_path, "alert_data")
EMAIL_ALERT_PATH = os.path.join(import_path, "alert_email")
IMG_DATA_FOLDER = os.path.join(import_path, "img_data")
ERROR_LOG = os.path.join(import_path, "error.log")
RUNTIME_LOG = os.path.join(import_path, "device_runtime.log")
NEW_JSON_PATH = os.path.join(import_path, "roi.json")

# Create necessary directories
os.makedirs(ALERT_DATA_FOLDER, exist_ok=True)
os.makedirs(EMAIL_ALERT_PATH, exist_ok=True)
os.makedirs(IMG_DATA_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# API configuration
API_KEY = ""
GEMINI_API = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Initialize models
yolo_model = YOLO("yolo11m.pt")  # Person detection
safety_model = YOLO("waaree_final.pt")  # Safety equipment detection
YOLO_CONFIDENCE_THRESHOLD = 0.7
SAFETY_CONFIDENCE_THRESHOLD = 0.01

# Safety equipment class mapping
SAFETY_CLASSES = {
    0: "blue_gloves",
    1: "cotton_gloves",
    2: "shoes",
    3: "ppe",
    4: "mask"
}

# Per-class confidence thresholds for safety equipment
SAFETY_CLASS_CONFIDENCE = {
    "blue_gloves": 0.05,
    "cotton_gloves": 0.01,
    "shoes": 0.01,
    "ppe": 0.2,
    "mask": 0.05
}

# Global state tracking
_crowd_counter = {}
_loitering_tracker = {}  # (cam_name): {'images': [], 'crops': [], 'bboxes': [], 'features': [], 'timestamps': []}
_downtime_tracker = {}  # (cam_name, roi_name): {'start_time': float, 'start_image': img, 'status': 'red'/'green', 'duration': 0}
_alert_cooldown = {}  # (cam_name, usecase): next_allowed_time

# Default scheduling frequency
DEFAULT_FREQ_SEC = 30

# ---------------------- CORE UTILITY FUNCTIONS ----------------------
def load_config():
    """Load configuration from JSON file"""
    try:
        with open(NEW_JSON_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

def capture_image(rtsp_url):
    """Capture image from RTSP stream"""
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise ValueError(f"Cannot open RTSP stream: {rtsp_url}")
        
        # Try to grab a few frames to stabilize
        for _ in range(5):
            cap.grab()
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise ValueError("Frame not captured")
        
        return frame
    except Exception as e:
        logging.error(f"Image capture error ({rtsp_url}): {e}")
        return None

def is_image_corrupted(img):
    """Check if captured image is corrupted or blank"""
    if img is None:
        return True
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check if image is too blurry or uniform
        if laplacian_var < 10:
            logging.warning("Image appears to be corrupted (low variance)")
            return True
        
        return False
    except Exception as e:
        logging.error(f"Error checking image corruption: {e}")
        return True

def save_alert(timestamp, cam_name, usecase, img, count, area_info=None):
    """Save alert data and image"""
    try:
        alert_type = usecase
        json_filename = f"{timestamp}_{alert_type}_{cam_name}.json"
        json_filepath1 = os.path.join(ALERT_DATA_FOLDER, json_filename)
        json_filepath2 = os.path.join(EMAIL_ALERT_PATH, json_filename)
        
        _, buffer = cv2.imencode(".jpg", img)
        img_str = base64.b64encode(buffer).decode("utf-8")

        camera_no = f"{cam_name}-{area_info}" if area_info else f"{cam_name}"
        
        entry = {
            "date_time": timestamp.replace("_", " "),
            "camera_no": camera_no,
            "area": area_info if area_info else "",
            "client": "Waaree",
            "alert_type": alert_type,
            "count": count,
            "image_byte_str": img_str
        }
        
        # Save JSON files only (no image file)
        with open(json_filepath1, "w") as f:
            json.dump(entry, f, indent=4)
        with open(json_filepath2, "w") as f:
            json.dump(entry, f, indent=4)
        
        logging.info(f"Alert saved: {alert_type} for {cam_name}")
    except Exception as e:
        logging.error(f"Error saving alert: {e}")

def log_runtime(cam_name, triggers):
    """Log runtime status of all use cases"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(RUNTIME_LOG, "a") as f:
            # f.write(f"{timestamp} - {cam_name}\n")
            for usecase, triggered in triggers.items():
                # f.write(f"{usecase}: {triggered}\n")
                print()
            # f.write("\n")
    except Exception as e:
        logging.error(f"Error logging runtime: {e}")

# ---------------------- GEMINI API FUNCTIONS ----------------------
def call_gemini_api(img_data, prompt):
    """Generic function to call Gemini API"""
    try:
        data = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_data}}
                ]
            }]
        }
        resp = requests.post(GEMINI_API, json=data, timeout=20)
        print(resp.text)
        if resp.status_code == 200:
            return resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip().lower()
        else:
            logging.error(f"Gemini API error: Status {resp.status_code}")
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
    return None

# ---------------------- USE CASE FUNCTIONS ----------------------
def process_crowd_detection(img, min_samples=4, eps=40):
    """Detect crowds using YOLO and DBSCAN clustering"""
    try:
        results = yolo_model(img, conf=YOLO_CONFIDENCE_THRESHOLD, classes=[0])
        person_centers = []
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                person_centers.append([cx, cy])
                
        if len(person_centers) < min_samples:
            return False, len(person_centers), None
            
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(person_centers)
        labels = clustering.labels_
        crowd_detected = (np.sum(labels != -1) >= min_samples)
        
        return crowd_detected, len(person_centers), labels
    except Exception as e:
        logging.error(f"Error in crowd detection: {e}")
        return False, 0, None

def extract_person_features(img_crop):
    """Extract feature vector from person crop for matching"""
    try:
        if img_crop is None or img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
            return None
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 128))
        feature_vector = resized.flatten() / 255.0
        return feature_vector
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def process_loitering(img, roi, cam_name):
    """
    Advanced loitering detection using feature matching across 3 frames.
    Captures frames over time and matches persons using cosine similarity.
    """
    try:
        results = yolo_model(img, conf=YOLO_CONFIDENCE_THRESHOLD, classes=[0], verbose=False)
        annotated_img = img.copy()
        
        # Handle ROI
        roi_points = None
        if roi and len(roi) > 0:
            roi_polygon = roi[0] if isinstance(roi[0], list) else roi
            roi_points = np.array(roi_polygon, dtype=np.int32)
        
        # Detect people in ROI
        people_in_roi = []
        crops = []
        bboxes = []
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Check if in ROI
                if roi_points is not None:
                    if cv2.pointPolygonTest(roi_points, person_center, False) < 0:
                        continue
                
                people_in_roi.append(person_center)
                person_crop = img[y1:y2, x1:x2]
                crops.append(person_crop)
                bboxes.append((x1, y1, x2, y2))
        
        # Initialize tracker for this camera if not exists
        if cam_name not in _loitering_tracker:
            _loitering_tracker[cam_name] = {
                'images': [],
                'crops': [],
                'bboxes': [],
                'features': [],
                'timestamps': []
            }
        
        tracker = _loitering_tracker[cam_name]
        current_time = time.time()
        
        # Extract features for current frame
        current_features = [extract_person_features(crop) for crop in crops]
        
        # Add current frame data
        tracker['images'].append(img.copy())
        tracker['crops'].append(crops)
        tracker['bboxes'].append(bboxes)
        tracker['features'].append(current_features)
        tracker['timestamps'].append(current_time)
        
        # Keep only last 3 frames
        if len(tracker['images']) > 3:
            tracker['images'].pop(0)
            tracker['crops'].pop(0)
            tracker['bboxes'].pop(0)
            tracker['features'].pop(0)
            tracker['timestamps'].pop(0)
        
        # Check if we have 3 frames captured over sufficient time (at least 4 minutes total)
        if len(tracker['images']) == 3:
            time_span = tracker['timestamps'][-1] - tracker['timestamps'][0]
            if time_span >= 240:  # 4 minutes
                # Perform matching across 3 frames
                loitering_detected = False
                loitering_boxes = []
                
                features1 = tracker['features'][0]
                features2 = tracker['features'][1]
                features3 = tracker['features'][2]
                bboxes1 = tracker['bboxes'][0]
                bboxes2 = tracker['bboxes'][1]
                bboxes3 = tracker['bboxes'][2]
                
                # Match persons across frames
                for i, feat1 in enumerate(features1):
                    if feat1 is None:
                        continue
                    
                    best_match_2, best_distance_2 = None, float("inf")
                    # Find matching person in frame 2
                    for j, feat2 in enumerate(features2):
                        if feat2 is None:
                            continue
                        distance_2 = cosine(feat1, feat2)
                        if distance_2 < best_distance_2:
                            best_distance_2, best_match_2 = distance_2, j
                    
                    # If match found in frame 2
                    if best_match_2 is not None and best_distance_2 < 0.3:
                        best_match_3, best_distance_3 = None, float("inf")
                        # Find matching person in frame 3
                        for k, feat3 in enumerate(features3):
                            if feat3 is None:
                                continue
                            distance_3 = cosine(features2[best_match_2], feat3)
                            if distance_3 < best_distance_3:
                                best_distance_3, best_match_3 = distance_3, k
                        
                        # If match found in all 3 frames
                        if best_match_3 is not None and best_distance_3 < 0.3:
                            loitering_detected = True
                            loitering_boxes.append((
                                bboxes1[i],
                                bboxes2[best_match_2],
                                bboxes3[best_match_3]
                            ))
                            logging.info(f"Loitering detected for person {i + 1} in {cam_name}")
                
                # If loitering detected, annotate all 3 images
                if loitering_detected:
                    annotated_images = [
                        tracker['images'][0].copy(),
                        tracker['images'][1].copy(),
                        tracker['images'][2].copy()
                    ]
                    
                    for (box1, box2, box3) in loitering_boxes:
                        for img_idx, box in enumerate([box1, box2, box3]):
                            x1, y1, x2, y2 = box
                            cv2.rectangle(annotated_images[img_idx], (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Clear tracker after detection
                    _loitering_tracker[cam_name] = {
                        'images': [],
                        'crops': [],
                        'bboxes': [],
                        'features': [],
                        'timestamps': []
                    }
                    
                    return True, len(loitering_boxes), annotated_images[-1]
                else:
                    # Reset tracker if no loitering found
                    _loitering_tracker[cam_name] = {
                        'images': [],
                        'crops': [],
                        'bboxes': [],
                        'features': [],
                        'timestamps': []
                    }
        
        # Return False - still collecting frames, no loitering detected yet
        return False, 0, annotated_img
        
    except Exception as e:
        logging.error(f"Error in loitering detection: {e}")
        return False, 0, img

def process_phone_detection(img):
    """Detect people using phones using YOLO + Gemini"""
    
    results = yolo_model(img, conf=YOLO_CONFIDENCE_THRESHOLD, classes=[0])
    annotated_img = img.copy()
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:  # Skip non-person detections
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_crop = img[y1:y2, x1:x2]
            
            # Use Gemini to detect phone usage
            _, buffer = cv2.imencode(".jpg", person_crop)
            img_data = base64.b64encode(buffer).decode("utf-8")
            prompt = (
                "Mobile Phone Detection Protocol:\n\n"
                "Identification Criteria:\n- At least 60% of phone must be clearly visible\n"
                "- Confidence Threshold: 90% or higher\n\n"
                "Strict Verification:\n- Confirm unambiguous device characteristics\n"
                "- Validate:\n* Substantial device boundaries\n* Clear user interaction\n"
                "* High-confidence visual markers\n\n"
                "Decision Rules:\n- Respond 'yes' ONLY if:\n* 60%+ of phone is visible\n"
                "* Confidence level reaches 90%\n- Default to 'no' if any uncertainty exists\n\n"
                "Prioritize precision over speculation."
            )
            
            if call_gemini_api(img_data, prompt) == "yes":
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_img, "Phone Usage", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return True, annotated_img
                
    return False, annotated_img

def process_blade_detection(img, roi=None):
    """
    Detect people carrying blades/knives using YOLO + Gemini
    Only processes persons inside the specified ROI
    Returns:
        detected (bool): Whether blade was detected
        annotated_img: Image with detections marked
    """
    results = yolo_model(img, conf=YOLO_CONFIDENCE_THRESHOLD, classes=[0])
    annotated_img = img.copy()
    h, w = img.shape[:2]
    
    # Draw ROI if specified
    if roi:
        roi_points = np.array(roi, dtype=np.int32)
        cv2.polylines(annotated_img, [roi_points], True, (0, 255, 0), 2)
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:  # Skip non-person detections
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Check if person is in ROI (if specified)
            if roi:
                roi_points = np.array(roi, dtype=np.int32)
                if cv2.pointPolygonTest(roi_points, person_center, False) < 0:
                    continue  # Skip persons outside ROI
            
            # Enlarge bbox by 20%
            box_width = x2 - x1
            box_height = y2 - y1
            enlarge_x = int(box_width * 0.2)
            enlarge_y = int(box_height * 0.2)
            
            enlarged_x1 = max(0, x1 - enlarge_x)
            enlarged_y1 = max(0, y1 - enlarge_y)
            enlarged_x2 = min(w, x2 + enlarge_x)
            enlarged_y2 = min(h, y2 + enlarge_y)
            
            # Extract enlarged person crop
            person_crop = img[enlarged_y1:enlarged_y2, enlarged_x1:enlarged_x2]
            
            if person_crop.size == 0:
                continue
            
            # Use Gemini to detect blade/knife
            _, buffer = cv2.imencode(".jpg", person_crop)
            img_data = base64.b64encode(buffer).decode("utf-8")
            prompt = (
                "Blade/Knife Detection Protocol:\n\n"
                "Identification Criteria:\n- At least 60% of blade/knife must be clearly visible\n"
                "- Confidence Threshold: 90% or higher\n\n"
                "Strict Verification:\n- Confirm unambiguous blade characteristics\n"
                "- Validate:\n* Clear blade shape and structure\n* Visible handle or grip\n* High-confidence visual markers\n\n"
                "Decision Rules:\n- Respond 'yes' ONLY if:\n* 60%+ of blade/knife is visible\n"
                "* Confidence level reaches 90%\n- Default to 'no' if any uncertainty exists\n\n"
                "Prioritize precision over speculation."
            )
            
            if call_gemini_api(img_data, prompt) == "yes":
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(annotated_img, (enlarged_x1, enlarged_y1), (enlarged_x2, enlarged_y2), (255, 0, 0), 1)
                cv2.putText(annotated_img, "Blade Detected", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return True, annotated_img
                
    return False, annotated_img

def process_downtime_monitoring(img, roi_configs, cam_name, frequency_sec):
    """
    Detect production downtime by monitoring red light color.
    Counts consecutive red frames and triggers alert when production resumes.
    Calculates actual downtime duration as: red_frame_count × frequency_sec.
    """
    try:
        if not roi_configs or len(roi_configs) == 0:
            return False, "No ROI", img, ""
        
        annotated_img = img.copy()
        downtime_detected = False
        detected_status = "Green"
        detected_roi_name = ""
        
        c=0
        # Process each ROI independently
        for roi_config in roi_configs:
            c += 1
            print("for----------------", c)
            roi_name = roi_config.get('name', 'Unknown')
            print("roi_name-",roi_name)        
            roi_points_list = roi_config.get('points', [])
            
            if not roi_points_list or len(roi_points_list) == 0:
                continue
                
            roi_points = roi_points_list[0] if isinstance(roi_points_list[0], list) else roi_points_list
            roi_polygon = np.array(roi_points, dtype=np.int32)
            
            x_coords = [p[0] for p in roi_points]
            y_coords = [p[1] for p in roi_points]
            x1, y1 = max(0, min(x_coords)), max(0, min(y_coords))
            x2, y2 = min(img.shape[1], max(x_coords)), min(img.shape[0], max(y_coords))
            
            roi = img[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            color_ranges = {
                "Red": [(0, 255, 100), (7, 255, 206)],
                "Green": [(40, 50, 50), (90, 255, 255)]
            }
            
            max_pixels = 0
            highlighted_color = "None"
            
            for color, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixels = cv2.countNonZero(mask)
                
                if pixels > max_pixels:
                    max_pixels = pixels
                    highlighted_color = color
            
            tracker_key = (cam_name, roi_name)
            
            # Initialize tracker if not exists
            if tracker_key not in _downtime_tracker:
                _downtime_tracker[tracker_key] = {
                    'start_image': None,
                    'status': 'Green',
                    'red_frame_count': 0
                }
                print(f"📍 New tracker created for: {tracker_key}")
            
            tracker = _downtime_tracker[tracker_key]
            
            # If color is Unknown, maintain previous state
            if highlighted_color == "None":
                highlighted_color = tracker['status']
                print(f"⚪ {roi_name}: Unknown color detected - Maintaining previous state: {highlighted_color}")
            
            if highlighted_color == "Red":
                # Downtime detected
                if tracker['status'] != 'Red':
                    # Transition from Green to Red - start tracking
                    tracker['start_image'] = img.copy()
                    tracker['status'] = 'Red'
                    tracker['red_frame_count'] = 1
                    logging.info(f"Downtime started for {cam_name} - {roi_name}")
                else:
                    # Continue tracking - increment red frame count
                    tracker['red_frame_count'] += 1
                
                print(f"🔴 {roi_name}: Red detected - Frame count: {tracker['red_frame_count']}")
                
                # Draw red border
                cv2.polylines(annotated_img, [roi_polygon], True, (0, 0, 255), 2)
                cv2.putText(annotated_img, f"{roi_name}: Downtime (frames: {tracker['red_frame_count']})", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Mark that this specific ROI has downtime
                downtime_detected = True
                detected_status = "Red"
                detected_roi_name = roi_name
                
            elif highlighted_color == "Green":
                # Production running - only trigger alert if there was a Red→Green transition
                if tracker['status'] == 'Red' and tracker['red_frame_count'] > 0:
                    # Transition from Red to Green - trigger alert ONLY ONCE
                    red_frames = tracker['red_frame_count']
                    
                    print(f"🚨 ALERT TRIGGER: {roi_name} had {red_frames} red frames, now turning GREEN")
                    
                    # Calculate actual downtime duration in seconds
                    downtime_duration_sec = red_frames * frequency_sec
                    
                    # Annotate the start image with downtime info
                    alert_img = tracker['start_image'].copy()
                    cv2.polylines(alert_img, [roi_polygon], True, (0, 0, 255), 3)
                    
                    # Add downtime duration text near ROI
                    duration_text = f"Downtime: {downtime_duration_sec}sec"
                    text_x = x1 + 100
                    text_y = y1 + 30
                    
                    # Add background rectangle for text
                    (text_w, text_h), _ = cv2.getTextSize(duration_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    cv2.rectangle(alert_img, (text_x - 5, text_y - text_h - 5), 
                                (text_x + text_w + 5, text_y + 5), (0, 0, 0), -1)
                    cv2.putText(alert_img, duration_text, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
                    # Save alert with duration
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    
                    # Prepare alert data
                    alert_type = f"downtime_{roi_name}"
                    json_filename = f"{timestamp}_{alert_type}_{cam_name}.json"
                    json_filepath1 = os.path.join(ALERT_DATA_FOLDER, json_filename)
                    json_filepath2 = os.path.join(EMAIL_ALERT_PATH, json_filename)
                    
                    _, buffer = cv2.imencode(".jpg", alert_img)
                    img_str = base64.b64encode(buffer).decode("utf-8")
                    
                    area_info = ""  # Get from camera config if needed
                    camera_no = f"{cam_name}-{area_info}" if area_info else f"{cam_name}"
                    
                    entry = {
                        "date_time": timestamp.replace("_", " "),
                        "camera_no": camera_no,
                        "area": area_info if area_info else "",
                        "client": "Waaree",
                        "alert_type": alert_type,
                        "count": f"{downtime_duration_sec}sec",  # Store duration in seconds
                        "image_byte_str": img_str
                    }
                    
                    with open(json_filepath1, "w") as f:
                        json.dump(entry, f, indent=4)
                    with open(json_filepath2, "w") as f:
                        json.dump(entry, f, indent=4)
                    
                    print(f"✅ Downtime alert saved for {cam_name} - {roi_name}: {downtime_duration_sec}sec ({red_frames} frames × {frequency_sec}sec)")
                    
                    # Set return values to indicate alert was triggered
                    downtime_detected = True
                    detected_status = "Alert_Triggered"
                    detected_roi_name = roi_name
                    
                    # Reset tracker
                    tracker['start_image'] = None
                    tracker['status'] = 'Green'
                    tracker['red_frame_count'] = 0
                else:
                    # Green state, no transition - just normal production
                    print(f"🟢 {roi_name}: Green (Production running)")
                
                # Draw green border
                cv2.polylines(annotated_img, [roi_polygon], True, (0, 255, 0), 2)
                cv2.putText(annotated_img, f"{roi_name}: Production", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(downtime_detected, detected_status, detected_roi_name)
        return downtime_detected, detected_status, annotated_img, detected_roi_name
    except Exception as e:
        logging.error(f"Error in downtime monitoring: {e}")
        return False, "Error", img, ""

def process_fire_exit(img, roi):
    """
    Check if fire exit is blocked using ROI + Gemini
    Returns:
        blocked (bool): Whether exit is blocked
        description (str): Description of blockage
        annotated_img: Image with annotations
    """
    if len(roi) < 4:
        return False, "", img
        
    # Create ROI mask
    roi_points = np.array(roi, dtype=np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    
    # Extract and annotate ROI
    roi_img = cv2.bitwise_and(img, img, mask=mask)
    annotated_img = roi_img.copy()
    cv2.polylines(annotated_img, [roi_points], True, (0, 255, 0), 2)
    
    # Check for blockage using Gemini
    _, buffer = cv2.imencode(".jpg", annotated_img)
    img_data = base64.b64encode(buffer).decode("utf-8")
    prompt = (
        "Fire Exit Safety Assessment Protocol:\n\n"
        "Context: This is a designated fire exit area that must remain clear at all times.\n\n"
        "Inspection Criteria:\n"
        "1. Area Clearance:\n"
        "   - Check for ANY objects blocking the marked area\n"
        "   - This includes people, equipment, boxes, furniture, etc.\n\n"
        "2. Access Path:\n"
        "   - Verify the exit path is completely unobstructed\n"
        "   - Look for partial blockages or narrow passages\n\n"
        "3. Safety Compliance:\n"
        "   - The entire marked area should be clear\n"
        "   - Even temporary objects are considered violations\n\n"
        "Response Format:\n"
        "- Respond 'blocked' if ANY obstruction is detected\n"
        "- Respond 'clear' if the area is completely free\n"
        "- Include brief description of blocking objects if found\n\n"
        "Priority: Safety-Critical Assessment"
    )
    
    response = call_gemini_api(img_data, prompt)
    if response and 'blocked' in response:
        description = response.split('\n')[0] if '\n' in response else response
        return True, description, annotated_img
        
    return False, "", annotated_img

def process_intrusion(img, roi=None, conf_threshold=0.5):
    """
    Detect unauthorized persons during specified hours
    Returns:
        detected (bool): Whether intrusion was detected
        count (int): Number of intruders
        annotated_img: Image with detections marked
    """
    results = yolo_model(img, conf=YOLO_CONFIDENCE_THRESHOLD, classes=[0])
    intruders_found = 0
    annotated_img = img.copy()
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:  # Skip non-person detections
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Check if person is in ROI (if specified)
            if roi:
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                roi_points = np.array(roi, dtype=np.int32)
                if cv2.pointPolygonTest(roi_points, person_center, False) < 0:
                    continue
            
            # Mark intruder
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_img, "INTRUDER", (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            intruders_found += 1
    
    # Draw ROI if specified
    if roi and intruders_found > 0:
        roi_points = np.array(roi, dtype=np.int32)
        cv2.polylines(annotated_img, [roi_points], True, (0, 255, 0), 2)
    
    return intruders_found > 0, intruders_found, annotated_img

def process_safety_equipment(img, enabled_items, use_cases=None):
    """
    Detect safety equipment violations using person-centered 640x640 crops
    Returns:
        detected (bool): Whether violations were found
        count (int): Number of violations
        annotated_img: Image with violations marked
        violations: List of violation details
    """
    print("\nStarting safety equipment detection process...")
    print("Step 1: Detecting persons in the frame")
    # First detect persons
    results = yolo_model(img, conf=YOLO_CONFIDENCE_THRESHOLD, classes=[0])
    person_boxes = []
    for r in results:
        person_boxes.extend([box for box in r.boxes if int(box.cls[0]) == 0])
    
    if not person_boxes:
        return False, 0, img, []
        
    violations = []
    annotated_img = img.copy()
    violation_count = 0
    h, w = img.shape[:2]
    
    print(f"Found {len(person_boxes)} persons in frame")
    
    # Get ROIs for each enabled safety item
    print("Step 2: Checking ROI configuration")
    item_rois = {}
    if use_cases:
        for use_case in ['mask_detection', 'shoes_detection', 'gloves_detection', 'ppe_detection']:
            item_key = use_case.split('_')[0]
            if enabled_items.get(item_key, False):
                # Accept either a single ROI or a list of ROIs (new format from JSON)
                use_case_rois = use_cases.get(use_case, {}).get('roi', [])
                if use_case_rois:
                    rois = []
                    # Handle new format: list of polygons [[x,y], [x,y], ...]
                    if isinstance(use_case_rois, list) and len(use_case_rois) > 0:
                        # Check if it's a list of polygons (each polygon is a list of [x,y] points)
                        if isinstance(use_case_rois[0], list) and len(use_case_rois[0]) > 0:
                            # If first element is a coordinate pair [x, y]
                            if isinstance(use_case_rois[0][0], (int, float)):
                                rois.append(np.array(use_case_rois, dtype=np.int32))
                            else:
                                # Multiple polygons
                                for polygon in use_case_rois:
                                    if polygon and len(polygon) > 0:
                                        rois.append(np.array(polygon, dtype=np.int32))
                    if rois:
                        item_rois[item_key] = rois
    
    print("Step 3: Processing individual persons")
    print("---------------------------------------")
    person_count = 0
    # Process each person
    for person_box in person_boxes:
        person_count += 1
        print(f"\nAnalyzing person {person_count}/{len(person_boxes)}")
        x1, y1, x2, y2 = map(int, person_box.xyxy[0].tolist())
        person_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # For each enabled safety item, check if person is inside any ROI for that item
        skip_person = False
        for item_key, enabled in enabled_items.items():
            if enabled and item_key in item_rois:
                rois = item_rois[item_key]
                # If person is not inside any ROI for this item, skip detection for this person
                if not any(cv2.pointPolygonTest(roi, person_center, False) >= 0 for roi in rois):
                    skip_person = True
                    break
        if skip_person:
            continue
        
        # Create 640x640 crop centered on person
        crop_size = 640
        crop_x1 = max(0, person_center[0] - crop_size // 2)
        crop_y1 = max(0, person_center[1] - crop_size // 2)
        crop_x2 = min(w, crop_x1 + crop_size)
        crop_y2 = min(h, crop_y1 + crop_size)
        
        # Adjust if crop goes out of bounds
        if crop_x2 - crop_x1 < crop_size:
            crop_x1 = max(0, crop_x2 - crop_size)
        if crop_y2 - crop_y1 < crop_size:
            crop_y1 = max(0, crop_y2 - crop_size)
        
        # Extract person-centered crop
        person_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if person_crop.size == 0:  # Skip if crop is empty
            continue
            
        # Calculate 20% enlarged person bbox (in crop coordinates)
        box_width = x2 - x1
        box_height = y2 - y1
        enlarge_x = int(box_width * 0.2)
        enlarge_y = int(box_height * 0.2)
        
        # Convert person bbox to crop coordinates
        person_x1_crop = x1 - crop_x1
        person_y1_crop = y1 - crop_y1
        person_x2_crop = x2 - crop_x1
        person_y2_crop = y2 - crop_y1
        
        # Calculate enlarged box in crop coordinates
        enlarged_x1_crop = max(0, person_x1_crop - enlarge_x)
        enlarged_y1_crop = max(0, person_y1_crop - enlarge_y)
        enlarged_x2_crop = min(crop_size, person_x2_crop + enlarge_x)
        enlarged_y2_crop = min(crop_size, person_y2_crop + enlarge_y)
        
        # Run safety detection on cropped area
        safety_results = safety_model(person_crop, conf=SAFETY_CONFIDENCE_THRESHOLD, verbose=True)
        print("Safety Detection Results:")
        print(f"Found {len(safety_results)} results")
        for r in safety_results:
            print(f"Number of detections: {len(r.boxes)}")
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                item_name = SAFETY_CLASSES.get(cls, "unknown")
                print(f"Detected {item_name} with confidence {conf:.2f}")
        
        # Track found safety items
        found_items = {
            'mask': False,
            'shoes': False,
            'gloves': False,
            'ppe': False
        }
        
        # Helper function to calculate IOU
        def calculate_iou(box1, box2):
            # box format: [x1, y1, x2, y2]
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            
            return intersection / union if union > 0 else 0
        
        # Process safety detections
        for r in safety_results:
            for det_box in r.boxes:
                cls = int(det_box.cls[0])
                conf = float(det_box.conf[0])
                if conf < SAFETY_CONFIDENCE_THRESHOLD:
                    continue
                
                # Get item name
                item_name = SAFETY_CLASSES.get(cls)
                
                # Get safety item bbox coordinates in crop space
                sx1, sy1, sx2, sy2 = map(int, det_box.xyxy[0].tolist())
                
                # Calculate IOU between safety item and enlarged person box
                iou = calculate_iou(
                    [sx1, sy1, sx2, sy2],
                    [enlarged_x1_crop, enlarged_y1_crop, enlarged_x2_crop, enlarged_y2_crop]
                )
                
                # If IOU > 0, the item overlaps with the person's enlarged box
                if iou > 0:
                    # Save the detected item with padding
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    pad = 20
                    save_x1 = max(0, sx1 - pad)
                    save_y1 = max(0, sy1 - pad)
                    save_x2 = min(person_crop.shape[1], sx2 + pad)
                    save_y2 = min(person_crop.shape[0], sy2 + pad)
                    
                    item_crop = person_crop[save_y1:save_y2, save_x1:save_x2]
                    if item_crop.size > 0:  # Check if crop is valid
                        filename = f"{timestamp}_{item_name}_{conf:.2f}_iou{iou:.2f}.jpg"
                        cv2.imwrite(os.path.join(IMG_DATA_FOLDER, filename), item_crop)
                    
                    # Mark item as found
                    if item_name in ['blue_gloves', 'cotton_gloves']:
                        found_items['gloves'] = True
                    elif item_name in found_items:
                        found_items[item_name] = True
        
        # Check for violations
        missing_items = []
        for item, enabled in enabled_items.items():
            if enabled and not found_items.get(item, False):
                missing_items.append(item)
                violation_count += 1
        
        if missing_items:
            violations.append({
                "bbox": [x1, y1, x2, y2],
                "missing_items": missing_items
            })
            # Draw violation boxes (red)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw 640x640 crop area (green)
            cv2.rectangle(annotated_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 1)
            
            # Convert enlarged box back to original image coordinates and draw it (blue)
            enlarged_x1_img = crop_x1 + enlarged_x1_crop
            enlarged_y1_img = crop_y1 + enlarged_y1_crop
            enlarged_x2_img = crop_x1 + enlarged_x2_crop
            enlarged_y2_img = crop_y1 + enlarged_y2_crop
            cv2.rectangle(annotated_img, 
                        (enlarged_x1_img, enlarged_y1_img), 
                        (enlarged_x2_img, enlarged_y2_img), 
                        (255, 0, 0), 1)
            
            text = "Missing: " + ", ".join(missing_items)
            cv2.putText(annotated_img, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return bool(violations), violation_count, annotated_img, violations
    

# ---------------------- MAIN CAMERA PROCESSING ----------------------
def process_camera(cam_config):
    """Main function to process all use cases for a camera"""
    cam_name = cam_config.get("cam_name", "Unknown")
    rtsp = cam_config.get("rtsp", "")
    area_info = cam_config.get("Area", "")
    use_cases = cam_config.get("use_cases", {})
    
    # Log enabled use cases
    enabled_usecases = [uc for uc, config in use_cases.items() if config.get('enabled', False)]
    # logging.info(f"Processing camera: {cam_name} | Area: {area_info} | Enabled use cases: {', '.join(enabled_usecases) if enabled_usecases else 'None'}")
    
    # Track use case triggers
    triggers = {
        "crowd": False,
        "loitering": False,
        "personusingphone": False,
        "intrusion": False,
        "fire_exit_blocked": False,
        "mask_detection": False,
        "shoes_detection": False,
        "gloves_detection": False,
        "ppe_detection": False,
        "blade": False,
        "downtime_monitoring": False
    }
    
    now = time.time()
    current_time = datetime.now().strftime("%H:%M")
    
    try:
        # Capture image from RTSP
        img = capture_image(rtsp)
        if img is None or is_image_corrupted(img):
            logging.warning(f"Failed to capture valid image from {cam_name}")
            return

        # Group 1: Person Detection Based Use Cases
        enabled_safety_items = {
            'mask': use_cases.get('mask_detection', {}).get('enabled', False),
            'shoes': use_cases.get('shoes_detection', {}).get('enabled', False),
            'gloves': use_cases.get('gloves_detection', {}).get('enabled', False),
            'ppe': use_cases.get('ppe_detection', {}).get('enabled', False)
        }

        person_detection_needed = (
            use_cases.get('intrusion', {}).get('enabled', False) or
            use_cases.get('personusingphone', {}).get('enabled', False) or
            use_cases.get('blade', {}).get('enabled', False) or
            any(enabled_safety_items.values())
        )
        
        if person_detection_needed:
            img = capture_image(rtsp)
            if img is not None and not is_image_corrupted(img):
                # Process intrusion detection
                #print("\n=== Checking Intrusion Detection ===")
                intrusion_config = use_cases.get('intrusion', {})
                if intrusion_config.get('enabled', False):
                    time_window = intrusion_config.get('time', ["00:00", "23:59"])
                    start_time, end_time = time_window
                    
                    # Check if current time is within intrusion monitoring window
                    is_monitoring_time = False
                    if start_time > end_time:  # Time window crosses midnight
                        is_monitoring_time = current_time >= start_time or current_time <= end_time
                    else:
                        is_monitoring_time = start_time <= current_time <= end_time
                        
                    if is_monitoring_time:
                        cooldown_key = (cam_name, "intrusion")
                        if _alert_cooldown.get(cooldown_key, 0) <= now:
                            detected, count, annotated_img = process_intrusion(
                                img,
                                roi=intrusion_config.get('roi', [])
                            )
                            if detected:
                                triggers["intrusion"] = True
                                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                save_alert(timestamp, cam_name, "intrusion", annotated_img, count)
                                _alert_cooldown[cooldown_key] = now + 300
                
                # Process phone usage detection
                #print("\n=== Checking Phone Usage ===")
                phone_config = use_cases.get('personusingphone', {})
                if phone_config.get('enabled', False):
                    cooldown_key = (cam_name, "phone")
                    if _alert_cooldown.get(cooldown_key, 0) <= now:
                        detected, annotated_img = process_phone_detection(img)
                        print(detected)
                        if detected:
                            triggers["personusingphone"] = True
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            save_alert(timestamp, cam_name, "phone", annotated_img, 1)
                            _alert_cooldown[cooldown_key] = now + 300
                
                # Process blade detection
                #print("\n=== Checking Blade Detection ===")
                blade_config = use_cases.get('blade', {})
                if blade_config.get('enabled', False):
                    cooldown_key = (cam_name, "blade")
                    if _alert_cooldown.get(cooldown_key, 0) <= now:
                        detected, annotated_img = process_blade_detection(
                            img,
                            roi=blade_config.get('roi', [])
                        )
                        if detected:
                            triggers["blade"] = True
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            save_alert(timestamp, cam_name, "blade", annotated_img, 1)
                            _alert_cooldown[cooldown_key] = now + 300
                
                # Process safety equipment violations
                if any(enabled_safety_items.values()):
                    #print("\n=== Checking Safety Equipment ===")
                    print(f"Enabled items: {[item for item, enabled in enabled_safety_items.items() if enabled]}")
                    cooldown_key = (cam_name, "safety")
                    if _alert_cooldown.get(cooldown_key, 0) <= now:
                        detected, count, annotated_img, violations = process_safety_equipment(
                            img, enabled_safety_items, use_cases
                        )
                        if detected:
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            
                            # Process each type of violation separately
                            for item_type in ['mask', 'shoes', 'gloves', 'ppe']:
                                # Get only violations for this specific item type
                                type_violations = [v for v in violations 
                                                if item_type in v["missing_items"]]
                                
                                if not type_violations:  # Skip if no violations of this type
                                    continue
                                    
                                # Create new image from original image for this violation type
                                violation_img = img.copy()
                                violation_count = len(type_violations)
                                
                                # Draw only the persons missing this specific item
                                for violation in type_violations:
                                    x1, y1, x2, y2 = violation["bbox"]
                                    cv2.rectangle(violation_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    text = f"Missing {item_type}"
                                    cv2.putText(violation_img, text, (x1, y1-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                                # Add a title to the image showing violation type and count
                                title = f"{item_type.upper()} Violations: {violation_count}"
                                # cv2.putText(violation_img, title, (10, 30),
                                #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                
                                # Save alert for this violation type
                                triggers[f"{item_type}_detection"] = True
                                alert_name = f"missing_{item_type}"
                                save_alert(timestamp, cam_name, alert_name, violation_img, violation_count)
                            
                            _alert_cooldown[cooldown_key] = now + 1800
        
        # Group 2: Crowd Detection
        #print("\n=== Checking Crowd Detection ===")
        crowd_config = use_cases.get('crowd', {})
        if crowd_config.get('enabled', False):
            cooldown_key = (cam_name, "crowd")
            if _alert_cooldown.get(cooldown_key, 0) <= now:
                if img is None:
                    img = capture_image(rtsp)
                if img is not None and not is_image_corrupted(img):
                    detected, count, _ = process_crowd_detection(img)
                    if detected:
                        _crowd_counter[cam_name] = _crowd_counter.get(cam_name, 0) + 1
                        print(_crowd_counter[cam_name])
                        if _crowd_counter[cam_name] >= 2:
                            triggers["crowd"] = True
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            save_alert(timestamp, cam_name, "crowd", img, count)
                            _alert_cooldown[cooldown_key] = now + 300
                            _crowd_counter[cam_name] = 0
                    else:
                        _crowd_counter[cam_name] = 0
        
        # Group 3: Loitering Detection
        #print("\n=== Checking Loitering Detection ===")
        loitering_config = use_cases.get('loitering', {})
        if loitering_config.get('enabled', False):
            logging.info(f"[{cam_name}] Checking: Loitering Detection")
            # Advanced loitering detection - no cooldown needed as it handles timing internally
            detected, count, loitering_img = process_loitering(
                img,
                roi=loitering_config.get('roi', []),
                cam_name=cam_name
            )
            triggers["loitering"] = bool(detected)
            logging.info(f"[{cam_name}] Loitering Detection Status: {triggers['loitering']}")
            if detected and loitering_img is not None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_alert(timestamp, cam_name, "loitering", loitering_img, count, area_info)
                logging.info(f"Loitering alert saved for {cam_name}")
        
        # Group 4: Fire Exit Monitoring
        #print("\n=== Checking Fire Exit Monitoring ===")
        fire_exit_config = use_cases.get('fire_exit', {})
        if fire_exit_config.get('enabled', False):
            cooldown_key = (cam_name, "fire_exit")
            if _alert_cooldown.get(cooldown_key, 0) <= now:
                roi = fire_exit_config.get('roi', [])
                if roi:
                    if img is None:
                        img = capture_image(rtsp)
                    if img is not None and not is_image_corrupted(img):
                        is_blocked, description, annotated_img = process_fire_exit(img, roi)
                        if is_blocked:
                            triggers["fire_exit_blocked"] = True
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            cv2.putText(annotated_img, f"Blocked: {description}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            save_alert(timestamp, cam_name, "fire_exit_blocked", annotated_img, 1)
                            _alert_cooldown[cooldown_key] = now + 300
        
        # Group 5: Downtime Monitoring
        print("\n=== Checking Downtime Monitoring ===")
        downtime_config = use_cases.get('downtime_monitoring', {})
        if downtime_config.get('enabled', False):
            cooldown_key = (cam_name, "downtime")
            if _alert_cooldown.get(cooldown_key, 0) <= now:
                roi_configs = downtime_config.get('roi', [])
                frequency_sec = downtime_config.get('frequency_sec', 30)
                if roi_configs:
                    if img is None:
                        img = capture_image(rtsp)
                    if img is not None and not is_image_corrupted(img):
                        detected, status, annotated_img, roi_name = process_downtime_monitoring(img, roi_configs, cam_name, frequency_sec)
                        # print(detected, status, roi_name)
        
    except Exception as e:
        logging.error(f"Error processing camera {cam_name}: {e}")
    
    # Log runtime status
    # log_runtime(cam_name, triggers)

# ---------------------- HARDCODED CONFIGURATION ----------------------
HARDCODED_CAMERAS = {
    "cam27": {
        "rtsp": "rtsp://admin:Vision24@192.168.1.248:554/Streaming/Channels/101",
        "cam_name": "1MLB-2_LINE-10_LAMINA",
        "Area": "MLB-2-Line-10",
        "use_cases": {
            "crowd": {
                "enabled": False,
                "frequency_sec": 30,
                "roi": [],
                "time": ["09:00", "18:00"],
                "models": ["yolo11m.pt"]
            },
            "loitering": {
                "enabled": False,
                "frequency_sec": 30,
                "roi": [],
                "time": ["00:00", "23:59"],
                "models": ["yolo11m.pt"]
            },
            "intrusion": {
                "enabled": False,
                "frequency_sec": 30,
                "roi": [],
                "time": ["10:00", "18:00"],
                "models": ["yolo11m.pt"]
            },
            "personusingphone": {
                "enabled": False,
                "frequency_sec": 10,
                "roi": [],
                "time": ["09:00", "18:00"],
                "models": ["yolo11m.pt"]
            },
            "fire_exit": {
                "enabled": False,
                "frequency_sec": 30,
                "roi": [],
                "time": ["00:00", "23:59"],
                "models": ["yolo11m.pt"],
                "check_blocking": True
            },
            "mask_detection": {
                "enabled": False,
                "frequency_sec": 30,
                "roi": [],
                "time": ["09:00", "18:00"],
                "models": ["yolo11m.pt", "waaree_final.pt"]
            },
            "shoes_detection": {
                "enabled": False,
                "frequency_sec": 30,
                "roi": [],
                "time": ["09:00", "18:00"],
                "models": ["yolo11m.pt", "waaree_final.pt"]
            },
            "gloves_detection": {
                "enabled": False,
                "frequency_sec": 30,
                "roi": [],
                "time": ["09:00", "18:00"],
                "models": ["yolo11m.pt", "waaree_final.pt"],
            },
            "ppe_detection": {
                "enabled": False,
                "frequency_sec": 30,
                "roi": [],
                "time": ["09:00", "18:00"],
                "models": ["yolo11m.pt", "waaree_final.pt"]
            },
            "downtime_monitoring": {
                "enabled": True,
                "frequency_sec": 5,
                "roi": [
                    {
                        "name": "test_dt",
                        "points": [ 
                            [[777,470],[777,497],[797,497],[797,470]]
                        ]
                    },
                    {
                        "name": "test_dtnon",
                        "points": [  
                            [[710,446],[710,470],[731,470],[731,446]]

                            [[740,446],[740,470],[758,470],[758,446]]
                            [[740, 369], [740, 393], [758, 393], [758, 369]]
                        ]
                    }
                ],
                "time": [
                    "00:00",
                    "23:59"
                ],
                "models": [
                    "yolo11m.pt",
                    "waaree_final.pt"
                ]
            }    
        }
    }
}

# ---------------------- SCHEDULER SETUP ----------------------
def schedule_device_tasks():
    """Set up periodic camera processing tasks"""
    device_config = {"cameras": HARDCODED_CAMERAS}
    for cam_name, cam_conf in device_config.get("cameras", {}).items():
        # Find minimum frequency among enabled use cases
        use_cases = cam_conf.get("use_cases", {})
        min_freq = None
        for v in use_cases.values():
            if v.get("enabled", False):
                freq = v.get("frequency_sec", DEFAULT_FREQ_SEC)
                if min_freq is None or freq < min_freq:
                    min_freq = freq
        
        # Schedule camera processing
        # print('sleep')
        freq_sec = min_freq if min_freq is not None else DEFAULT_FREQ_SEC
        schedule.every(freq_sec).seconds.do(process_camera, cam_conf)
# [[654,410],[658,430],[680,428],[672,404]],
# ---------------------- MAIN LOOP ----------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SCHEDULER STARTING...")
    print("=" * 60)
    schedule_device_tasks()
    
    # Print scheduled jobs
    print("\nScheduled Jobs:")
    print("-" * 60)
    for job in schedule.jobs:
        # print(f"Job: {job.job_func.func.__name__}")
        # print(f"Interval: Every {job.interval} {job.unit}")
        # print(f"Next run: {job.next_run}")
        print("-" * 60)
    
    last_run_check = time.time()
    sleep_counter = 0
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check and run pending jobs
            pending_jobs = [job for job in schedule.jobs if job.should_run]
            if pending_jobs:
                # print(f"\n[{current_time}] RUNNING {len(pending_jobs)} JOB(S)...")
                for job in pending_jobs:
                    # print(f"  → Executing: {job.job_func.func.__name__}")
                    print()
                
                schedule.run_pending()
                
                # Print next scheduled run times
                # print(f"\n[{current_time}] NEXT SCHEDULED RUNS:")
                for job in schedule.jobs:
                    print(f"  → {job.job_func.func.__name__}: {job.next_run}")
            else:
                sleep_counter += 1
                if sleep_counter % 60 == 0:  # Print every 60 seconds
                    # print(f"[{current_time}] SLEEPING... Next scheduled run:")
                    for job in schedule.jobs:
                        time_until = (job.next_run - datetime.now()).total_seconds()
                        if time_until > 0:
                            print(f"  → {job.job_func.func.__name__}: in {int(time_until)}s @ {job.next_run}")
                    sleep_counter = 0
            
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Scheduler error: {e}")
            print(f"❌ ERROR: {e}")

