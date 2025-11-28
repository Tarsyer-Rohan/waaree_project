""" 
V2 - 26/11/25
"""

import cv2
import numpy as np
import os
import json
import base64
import logging
import schedule
import time
import requests
import zmq
import pickle
from datetime import datetime
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

# ---------------------- CONFIG AND PATHS ----------------------
# Set up directory paths
import_path = os.path.dirname(os.path.abspath(__file__))
ALERT_DATA_FOLDER = "/tmp/alerts"
EMAIL_ALERT_PATH = os.path.join(import_path, "alert_email")
ERROR_LOG = "/tmp/error.log"
RUNTIME_LOG = "/tmp/device_runtime.log"
CONFIG_JSON_PATH = os.path.join(import_path, "all_cameras_config.json")
CAMERA_ANGLE_REF_DIR = os.path.join(import_path, "camera_angle")

# Create necessary directories
os.makedirs(ALERT_DATA_FOLDER, exist_ok=True)
os.makedirs(EMAIL_ALERT_PATH, exist_ok=True)
os.makedirs(CAMERA_ANGLE_REF_DIR, exist_ok=True)

# Configure logging - Only errors go to error.log, info to console
error_handler = logging.FileHandler(ERROR_LOG)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[error_handler, console_handler]
)

# API configuration
API_KEY = "AIzaSyCFfOYXv6YWvzyoNPVuSY43tpVHTlGBkEg"  #"AIzaSyCcWQX3S56uG_puWL9dOvLtU0iJl5g7UyY"
GEMINI_API = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ZMQ Client Configuration
PERSON_DETECTION_SERVER = "tcp://192.168.105.5:7771"
SAFETY_DETECTION_SERVER = "tcp://192.168.105.5:7772"

# Initialize ZMQ context and sockets
zmq_context = zmq.Context()
person_socket = None
safety_socket = None

def init_zmq_clients():
    """Initialize ZMQ client sockets"""
    global person_socket, safety_socket
    try:
        person_socket = zmq_context.socket(zmq.REQ)
        person_socket.connect(PERSON_DETECTION_SERVER)
        person_socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        person_socket.setsockopt(zmq.SNDTIMEO, 30000)
        logging.info(f"Connected to person detection server at {PERSON_DETECTION_SERVER}")
        
        safety_socket = zmq_context.socket(zmq.REQ)
        safety_socket.connect(SAFETY_DETECTION_SERVER)
        safety_socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        safety_socket.setsockopt(zmq.SNDTIMEO, 30000)
        logging.info(f"Connected to safety detection server at {SAFETY_DETECTION_SERVER}")
    except Exception as e:
        logging.error(f"Error initializing ZMQ clients: {e}")
        raise

def call_person_detection(image, conf_threshold=0.6, classes=[0], verbose=False):
    """
    Call person detection server via ZMQ
    
    Args:
        image: numpy array (BGR image)
        conf_threshold: Confidence threshold
        classes: List of class IDs to detect
        verbose: Verbose output
        
    Returns:
        List of detections
    """
    global person_socket
    try:
        request = {
            "image": image,
            "conf_threshold": conf_threshold,
            "classes": classes,
            "verbose": verbose
        }
        
        # Send request
        person_socket.send(pickle.dumps(request))
        
        # Receive response
        response = pickle.loads(person_socket.recv())
        
        if response.get("error"):
            logging.error(f"Person detection error: {response['error']}")
            return []
        
        return response.get("detections", [])
    except zmq.error.Again:
        logging.error("Person detection server timeout - no response received")
        # Reconnect socket
        person_socket.close()
        person_socket = zmq_context.socket(zmq.REQ)
        person_socket.connect(PERSON_DETECTION_SERVER)
        person_socket.setsockopt(zmq.RCVTIMEO, 30000)
        person_socket.setsockopt(zmq.SNDTIMEO, 30000)
        return []
    except Exception as e:
        logging.error(f"Error calling person detection server: {e}")
        return []

def call_safety_detection(image, conf_threshold=0.05, verbose=False):
    """
    Call safety equipment detection server via ZMQ
    
    Args:
        image: numpy array (person crop)
        conf_threshold: Confidence threshold
        verbose: Verbose output
        
    Returns:
        Dictionary with detections and found_items
    """
    global safety_socket
    try:
        request = {
            "image": image,
            "conf_threshold": conf_threshold,
            "verbose": verbose
        }
        
        # Send request
        safety_socket.send(pickle.dumps(request))
        
        # Receive response
        response = pickle.loads(safety_socket.recv())
        
        if response.get("error"):
            logging.error(f"Safety detection error: {response['error']}")
            return {"detections": [], "found_items": {}}
        
        return response
    except zmq.error.Again:
        logging.error("Safety detection server timeout - no response received")
        # Reconnect socket
        safety_socket.close()
        safety_socket = zmq_context.socket(zmq.REQ)
        safety_socket.connect(SAFETY_DETECTION_SERVER)
        safety_socket.setsockopt(zmq.RCVTIMEO, 30000)
        safety_socket.setsockopt(zmq.SNDTIMEO, 30000)
        return {"detections": [], "found_items": {}}
    except Exception as e:
        logging.error(f"Error calling safety detection server: {e}")
        return {"detections": [], "found_items": {}}

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
_downtime_tracker = {}  # (cam_name, roi_name): {'start_image': img, 'status': 'Red'/'Green', 'red_frame_count': 0}
_alert_cooldown = {}  # (cam_name, usecase): next_allowed_time
_camera_angle_refs = {}  # cam_name: reference_image_path

# Default scheduling frequency
DEFAULT_FREQ_SEC = 30

# ---------------------- CORE UTILITY FUNCTIONS ----------------------
def load_config():
    """Load configuration from JSON file"""
    try:
        if not os.path.exists(CONFIG_JSON_PATH):
            logging.error(f"Config file not found: {CONFIG_JSON_PATH}")
            return []
        
        with open(CONFIG_JSON_PATH, "r") as f:
            config = json.load(f)
            logging.info(f"Loaded configuration for {len(config)} cameras")
            return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return []

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
        
        # Save image and JSON files
        # cv2.imwrite(os.path.join(ALERT_DATA_FOLDER, f"{timestamp}_{alert_type}_{cam_name}.jpg"), img)
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
            f.write(f"{timestamp} - {cam_name}\n")
            for usecase, triggered in triggers.items():
                f.write(f"{usecase}: {triggered}\n")
            f.write("\n")
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
        # print(resp.text)
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
        detections = call_person_detection(img, conf_threshold=YOLO_CONFIDENCE_THRESHOLD, classes=[0], verbose=True)
        person_centers = []
        
        for det in detections:
            cx, cy = det["center"]
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
        detections = call_person_detection(img, conf_threshold=YOLO_CONFIDENCE_THRESHOLD, classes=[0], verbose=False)
        annotated_img = img.copy()
        
        # Handle ROI
        roi_points = None
        if roi and len(roi) > 0:
            roi_polygon = roi[0] if isinstance(roi[0], list) else roi
            roi_points = np.array(roi_polygon, dtype=np.int32)
            #cv2.polylines(annotated_img, [roi_points], True, (0, 255, 0), 2)
        
        # Detect people in ROI
        people_in_roi = []
        crops = []
        bboxes = []
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            person_center = tuple(det["center"])
            
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
                            # cv2.putText(annotated_images[img_idx], "Loitering", (x1, y1 - 10),
                            #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
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
    try:
        detections = call_person_detection(img, conf_threshold=YOLO_CONFIDENCE_THRESHOLD, classes=[0], verbose=True)
        annotated_img = img.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            person_crop = img[y1:y2, x1:x2]
            
            _, buffer = cv2.imencode(".jpg", person_crop)
            img_data = base64.b64encode(buffer).decode("utf-8")
            prompt = (
                "Mobile Phone Detection Protocol:\n\n"
                "Identification Criteria:\n- At least 60% of phone must be clearly visible\n"
                "- Confidence Threshold: 90% or higher\n\n"
                "Decision Rules:\n- Respond 'yes' ONLY if:\n* 60%+ of phone is visible\n"
                "* Confidence level reaches 90%\n- Default to 'no' if any uncertainty exists"
            )
            
            if call_gemini_api(img_data, prompt) == "yes":
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(annotated_img, "Phone Usage", (x1, y1-10),
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return True, annotated_img
                    
        return False, annotated_img
    except Exception as e:
        logging.error(f"Error in phone detection: {e}")
        return False, img

def process_blade_detection(img, roi=None):
    """Detect people carrying blades/knives using YOLO + Gemini"""
    try:
        detections = call_person_detection(img, conf_threshold=YOLO_CONFIDENCE_THRESHOLD, classes=[0], verbose=True)
        annotated_img = img.copy()
        h, w = img.shape[:2]
        
        # Handle ROI format
        if roi and len(roi) > 0:
            roi_polygon = roi[0] if isinstance(roi[0], list) else roi
            roi_points = np.array(roi_polygon, dtype=np.int32)
            # cv2.polylines(annotated_img, [roi_points], True, (0, 255, 0), 2)
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            person_center = tuple(det["center"])
            
            if roi and len(roi) > 0:
                roi_polygon = roi[0] if isinstance(roi[0], list) else roi
                roi_points = np.array(roi_polygon, dtype=np.int32)
                if cv2.pointPolygonTest(roi_points, person_center, False) < 0:
                    continue
            
            # Enlarge bbox by 20%
            box_width = x2 - x1
            box_height = y2 - y1
            enlarge_x = int(box_width * 0.2)
            enlarge_y = int(box_height * 0.2)
            
            enlarged_x1 = max(0, x1 - enlarge_x)
            enlarged_y1 = max(0, y1 - enlarge_y)
            enlarged_x2 = min(w, x2 + enlarge_x)
            enlarged_y2 = min(h, y2 + enlarge_y)
            
            person_crop = img[enlarged_y1:enlarged_y2, enlarged_x1:enlarged_x2]
            
            if person_crop.size == 0:
                continue
            
            _, buffer = cv2.imencode(".jpg", person_crop)
            img_data = base64.b64encode(buffer).decode("utf-8")
            # prompt1 = (
            #     "Blade/Knife Detection Protocol:\n\n"
            #     "Respond 'yes' ONLY if 60%+ of blade/knife is visible with 90% confidence\n"
            #     "Default to 'no' if any uncertainty exists"
            # )
            # prompt2 = (
            #     "Naked Blade Detection Protocol:\n\n"
            #     "Respond 'yes' ONLY if the person is holding a blade, cutter, or sharp metal object "
            #     "directly by the exposed metal (no handle or grip present), and at least 5% of the metal "
            #     "blade surface is clearly visible. The detection confidence must also be 50% or higher.\n\n"
            #     "If the object appears to have a handle, casing, grip, or if visibility or confidence "
            #     "is below 50%, respond strictly with 'no'."
            # )
            prompt = (
                "Naked Blade Detection Protocol:\n\n"
                "Respond 'yes' if the person is holding a blade, cutter, or sharp metal object and the exposed metal part is clearly visible (even if there is a small handle or grip). At least 3% of the metal blade surface should be visible, and detection confidence must be 40% or higher.\n\n"
                "If the object is mostly covered by a handle, casing, or grip, or if visibility/confidence is very low, respond with 'no'."
            )
            
            if call_gemini_api(img_data, prompt) == "yes":
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                return True, annotated_img
                    
        return False, annotated_img
    except Exception as e:
        logging.error(f"Error in blade detection: {e}")
        return False, img

def process_downtime_monitoring(img, roi_configs, cam_name, frequency_sec, area_info=None):
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
        
        # Process each ROI independently
        for roi_config in roi_configs:
            roi_name = roi_config.get('name', 'Unknown')
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
            
            tracker = _downtime_tracker[tracker_key]
            
            # If color is Unknown, maintain previous state
            if highlighted_color == "None":
                highlighted_color = tracker['status']
            
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
                
                # Draw red border
                cv2.polylines(annotated_img, [roi_polygon], True, (0, 0, 255), 2)
                downtime_detected = True
                detected_status = "Red"
                detected_roi_name = roi_name
                
            elif highlighted_color == "Green":
                # Production running - only trigger alert if there was a Red→Green transition
                if tracker['status'] == 'Red' and tracker['red_frame_count'] > 0:
                    # Transition from Red to Green - trigger alert
                    red_frames = tracker['red_frame_count']
                    
                    # Calculate actual downtime duration in seconds
                    downtime_duration_sec = red_frames * frequency_sec
                    
                    # Annotate the start image with downtime info
                    alert_img = tracker['start_image'].copy()
                    cv2.polylines(alert_img, [roi_polygon], True, (0, 0, 255), 3)
                    
                    # Add downtime duration text near ROI
                    duration_text = f"Downtime:{downtime_duration_sec}sec"
                    text_x = x1 - 10
                    text_y = y1 - 30
                    
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
                    
                    area_info = area_info  # Get from camera config if needed
                    camera_no = f"{cam_name}-{area_info}" if area_info else f"{cam_name}"
                    
                    entry = {
                        "date_time": timestamp.replace("_", " "),
                        "camera_no": camera_no,
                        "area": area_info if area_info else "",
                        "client": "Waaree",
                        "alert_type": alert_type,
                        "count": downtime_duration_sec,  
                        "image_byte_str": img_str
                    }
                    
                    with open(json_filepath1, "w") as f:
                        json.dump(entry, f, indent=4)
                    with open(json_filepath2, "w") as f:
                        json.dump(entry, f, indent=4)
                    
                    logging.info(f"Downtime alert saved for {cam_name} - {roi_name}: {downtime_duration_sec}sec ({red_frames} frames × {frequency_sec}sec)")
                    
                    # Set return values to indicate alert was triggered
                    downtime_detected = True
                    detected_status = "Alert_Triggered"
                    detected_roi_name = roi_name
                    
                    # Reset tracker
                    tracker['start_image'] = None
                    tracker['status'] = 'Green'
                    tracker['red_frame_count'] = 0
                
                # Draw green border
                cv2.polylines(annotated_img, [roi_polygon], True, (0, 255, 0), 2)
            
        return downtime_detected, detected_status, annotated_img, detected_roi_name
    except Exception as e:
        logging.error(f"Error in downtime monitoring: {e}")
        return False, "Error", img, ""

def check_camera_angle(ref_img, current_img, min_matches=20, deviation_threshold=2.5):
    """
    Compare a reference image and a current image to check for camera angle changes.
    """
    try:
        # Convert both images to grayscale
        gray1 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        
        # ORB feature detector and descriptor
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        # Ensure features are found in both images
        if des1 is None or des2 is None:
            return 'NO_CHANGE', 'Could not find features in one image.', None

        # Brute-force matching with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Check if enough matches exist
        if len(matches) < min_matches:
            reason = f"Found only {len(matches)} matches (min is {min_matches})."
            return 'NO_CHANGE', reason, None

        # Sort and select best matches
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:100]

        # Extract corresponding points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return 'NO_CHANGE', 'Homography computation failed.', None

        # Compute deviation from identity
        identity_matrix = np.identity(3)
        deviation = np.linalg.norm(H - identity_matrix)

        # Determine if deviation exceeds threshold
        if deviation > deviation_threshold:
            status = 'ANGLE_CHANGED'
            reason = f"Deviation ({deviation:.2f}) exceeds threshold ({deviation_threshold:.2f})."
        else:
            status = 'NO_CHANGE'
            reason = f"Deviation ({deviation:.2f}) is within threshold."
            
        return status, reason, deviation
    except Exception as e:
        logging.error(f"Error in camera angle check: {e}")
        return 'ERROR', str(e), None

def process_camera_angle_check(img, cam_name, min_matches=30, deviation_threshold=3.5):
    """
    Check if camera angle has changed by comparing with reference image.
    If reference doesn't exist, create it from current frame.
    """
    try:
        ref_img_path = os.path.join(CAMERA_ANGLE_REF_DIR, f"{cam_name}_ref.png")
        
        # If reference image does not exist, save current frame as reference
        if not os.path.exists(ref_img_path):
            cv2.imwrite(ref_img_path, img)
            _camera_angle_refs[cam_name] = ref_img_path
            return False, img  # No check on first frame
        
        # Load reference image
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            logging.error(f"Failed to load reference image: {ref_img_path}")
            return False, img
        
        # Compare current frame with reference
        status, reason, deviation = check_camera_angle(ref_img, img, min_matches, deviation_threshold)
        
        if status == 'ANGLE_CHANGED':
            annotated_img = img.copy()
            # cv2.putText(annotated_img, "CAMERA ANGLE CHANGED", (50, 50),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return True, annotated_img, deviation
        else:
            return False, img, None
            
    except Exception as e:
        logging.error(f"Error in camera angle check for {cam_name}: {e}")
        return False, img, None
def process_intrusion(img, roi=None):
    """Detect unauthorized persons"""
    try:
        detections = call_person_detection(img, conf_threshold=YOLO_CONFIDENCE_THRESHOLD, classes=[0], verbose=True)
        intruders_found = 0
        annotated_img = img.copy()
        
        # Handle ROI format
        if roi and len(roi) > 0:
            roi_polygon = roi[0] if isinstance(roi[0], list) else roi
            roi_points = np.array(roi_polygon, dtype=np.int32)
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            
            if roi and len(roi) > 0:
                person_center = tuple(det["center"])
                if cv2.pointPolygonTest(roi_points, person_center, False) < 0:
                    continue
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.putText(annotated_img, "INTRUDER", (x1, y1-10),
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            intruders_found += 1
        
        # if roi and len(roi) > 0 and intruders_found > 0:
            #cv2.polylines(annotated_img, [roi_points], True, (0, 255, 0), 2)
        
        return intruders_found > 0, intruders_found, annotated_img
    except Exception as e:
        logging.error(f"Error in intrusion detection: {e}")
        return False, 0, img

def process_safety_equipment(img, enabled_items, use_cases=None):
    """Detect safety equipment violations"""
    try:
        detections = call_person_detection(img, conf_threshold=YOLO_CONFIDENCE_THRESHOLD, classes=[0], verbose=True)
        
        if not detections:
            return False, 0, img, []
            
        violations = []
        annotated_img = img.copy()
        violation_count = 0
        h, w = img.shape[:2]
        
        # Get ROIs for each enabled safety item
        item_rois = {}
        if use_cases:
            for use_case in ['mask_detection', 'shoes_detection', 'gloves_detection', 'ppe_detection']:
                item_key = use_case.split('_')[0]
                if enabled_items.get(item_key, False):
                    use_case_rois = use_cases.get(use_case, {}).get('roi', [])
                    if use_case_rois:
                        rois = []
                        if isinstance(use_case_rois, list) and len(use_case_rois) > 0:
                            if isinstance(use_case_rois[0], list) and len(use_case_rois[0]) > 0:
                                if isinstance(use_case_rois[0][0], (int, float)):
                                    rois.append(np.array(use_case_rois, dtype=np.int32))
                                else:
                                    for polygon in use_case_rois:
                                        if polygon and len(polygon) > 0:
                                            rois.append(np.array(polygon, dtype=np.int32))
                        if rois:
                            item_rois[item_key] = rois
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            person_center = tuple(det["center"])
            
            # Calculate bbox dimensions
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Skip if bbox width is more than height (incorrect orientation)
            if bbox_width > bbox_height:
                continue
            
            # Skip if height is not at least 1.5 times the width (person too far/small)
            if bbox_height < 1.5 * bbox_width:
                continue

            # Check if person is in ROI
            skip_person = False
            for item_key, enabled in enabled_items.items():
                if enabled and item_key in item_rois:
                    rois = item_rois[item_key]
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
            
            if crop_x2 - crop_x1 < crop_size:
                crop_x1 = max(0, crop_x2 - crop_size)
            if crop_y2 - crop_y1 < crop_size:
                crop_y1 = max(0, crop_y2 - crop_size)
            
            person_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if person_crop.size == 0:
                continue
                
            box_width = x2 - x1
            box_height = y2 - y1
            enlarge_x = int(box_width * 0.2)
            enlarge_y = int(box_height * 0.2)
            
            person_x1_crop = x1 - crop_x1
            person_y1_crop = y1 - crop_y1
            person_x2_crop = x2 - crop_x1
            person_y2_crop = y2 - crop_y1
            
            enlarged_x1_crop = max(0, person_x1_crop - enlarge_x)
            enlarged_y1_crop = max(0, person_y1_crop - enlarge_y)
            enlarged_x2_crop = min(crop_size, person_x2_crop + enlarge_x)
            enlarged_y2_crop = min(crop_size, person_y2_crop + enlarge_y)
            
            # Call safety detection server
            safety_response = call_safety_detection(person_crop, conf_threshold=SAFETY_CONFIDENCE_THRESHOLD, verbose=True)
            safety_detections = safety_response.get("detections", [])
            
            found_items = {
                'mask': False,
                'shoes': False,
                'gloves': False,
                'ppe': False
            }
            
            def calculate_iou(box1, box2):
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
            
            for safety_det in safety_detections:
                item_name = safety_det["class_name"]
                confidence = safety_det.get("confidence", 0)
                sx1, sy1, sx2, sy2 = safety_det["bbox"]
                
                # Check if confidence meets the per-class threshold
                required_conf = SAFETY_CLASS_CONFIDENCE.get(item_name, 0.05)
                if confidence < required_conf:
                    continue
                
                iou = calculate_iou(
                    [sx1, sy1, sx2, sy2],
                    [enlarged_x1_crop, enlarged_y1_crop, enlarged_x2_crop, enlarged_y2_crop]
                )
                
                if iou > 0:
                    if item_name in ['blue_gloves', 'cotton_gloves']:
                        found_items['gloves'] = True
                    elif item_name in found_items:
                        found_items[item_name] = True
            
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
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # text = "Missing: " + ", ".join(missing_items)
                # cv2.putText(annotated_img, text, (x1, y1-10),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return bool(violations), violation_count, annotated_img, violations
    except Exception as e:
        logging.error(f"Error in safety equipment detection: {e}")
        return False, 0, img, []

# ---------------------- MAIN CAMERA PROCESSING ----------------------
def process_camera(cam_config):
    """Main function to process all use cases for a camera"""
    cam_name = cam_config.get("cam_name", "Unknown")
    rtsp = cam_config.get("rtsp", "")
    area_info = cam_config.get("Area", "")
    use_cases = cam_config.get("use_cases", {})
    
    # Log enabled use cases
    enabled_usecases = [uc for uc, config in use_cases.items() if config.get('enabled', False)]
    logging.info(f"Processing camera: {cam_name} | Area: {area_info} | Enabled use cases: {', '.join(enabled_usecases) if enabled_usecases else 'None'}")
    
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
        "downtime_monitoring": False,
        "camera_angle_check": False
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
            # Process intrusion detection
            intrusion_config = use_cases.get('intrusion', {})
            if intrusion_config.get('enabled', False):
                logging.info(f"[{cam_name}] Checking: Intrusion Detection")
                detected = False
                time_window = intrusion_config.get('time', ["00:00", "23:59"])
                start_time, end_time = time_window
                is_monitoring_time = False
                if start_time > end_time:
                    is_monitoring_time = current_time >= start_time or current_time <= end_time
                else:
                    is_monitoring_time = start_time <= current_time <= end_time
                if is_monitoring_time:
                    cooldown_key = (cam_name, "intrusion")
                    if _alert_cooldown.get(cooldown_key, 0) <= now:
                        # Set cooldown immediately to prevent duplicate processing
                        _alert_cooldown[cooldown_key] = now + 300
                        detected, count, annotated_img = process_intrusion(
                            img,
                            roi=intrusion_config.get('roi', [])
                        )
                        if detected:
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            save_alert(timestamp, cam_name, "intrusion", annotated_img, count, area_info)
                triggers["intrusion"] = bool(detected)
                logging.info(f"[{cam_name}] Intrusion Detection Status: {triggers['intrusion']}")

            # Process phone usage detection
            phone_config = use_cases.get('personusingphone', {})
            if phone_config.get('enabled', False):
                logging.info(f"[{cam_name}] Checking: Phone Usage Detection")
                detected = False
                cooldown_key = (cam_name, "phone")
                if _alert_cooldown.get(cooldown_key, 0) <= now:
                    # Set cooldown immediately to prevent duplicate processing
                    _alert_cooldown[cooldown_key] = now + 1000
                    detected, annotated_img = process_phone_detection(img)
                    if detected:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        save_alert(timestamp, cam_name, "phone", annotated_img, 1, area_info)
                triggers["personusingphone"] = bool(detected)
                logging.info(f"[{cam_name}] Phone Usage Detection Status: {triggers['personusingphone']}")

            # Process blade detection
            blade_config = use_cases.get('blade', {})
            if blade_config.get('enabled', False):
                logging.info(f"[{cam_name}] Checking: Blade Detection")
                detected = False
                cooldown_key = (cam_name, "blade")
                if _alert_cooldown.get(cooldown_key, 0) <= now:
                    # Set cooldown immediately to prevent duplicate processing
                    _alert_cooldown[cooldown_key] = now + 300
                    detected, annotated_img = process_blade_detection(
                        img,
                        roi=blade_config.get('roi', [])
                    )
                    if detected:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        save_alert(timestamp, cam_name, "blade", annotated_img, 1, area_info)
                triggers["blade"] = bool(detected)
                logging.info(f"[{cam_name}] Blade Detection Status: {triggers['blade']}")

            # Process safety equipment violations
            if any(enabled_safety_items.values()):
                logging.info(f"[{cam_name}] Checking: Safety Equipment ({', '.join([k for k, v in enabled_safety_items.items() if v])})")
                detected = False
                cooldown_key = (cam_name, "safety")
                if _alert_cooldown.get(cooldown_key, 0) <= now:
                    # Set cooldown immediately to prevent duplicate processing
                    _alert_cooldown[cooldown_key] = now + 1800  # 30 minutes cooldown
                    detected, count, annotated_img, violations = process_safety_equipment(
                        img, enabled_safety_items, use_cases
                    )
                    if detected:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        for item_type in ['mask', 'shoes', 'gloves', 'ppe']:
                            type_violations = [v for v in violations 
                                            if item_type in v["missing_items"]]
                            if not type_violations:
                                continue
                            violation_img = img.copy()
                            violation_count = len(type_violations)
                            for violation in type_violations:
                                x1, y1, x2, y2 = violation["bbox"]
                                cv2.rectangle(violation_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            triggers[f"{item_type}_detection"] = bool(type_violations)
                            alert_name = f"missing_{item_type}"
                            save_alert(timestamp, cam_name, alert_name, violation_img, violation_count, area_info)
                logging.info(f"[{cam_name}] Safety Equipment Status: mask={triggers['mask_detection']}, shoes={triggers['shoes_detection']}, gloves={triggers['gloves_detection']}, ppe={triggers['ppe_detection']}")

        # Group 2: Crowd Detection
        crowd_config = use_cases.get('crowd', {})
        if crowd_config.get('enabled', False):
            logging.info(f"[{cam_name}] Checking: Crowd Detection")
            detected = False
            cooldown_key = (cam_name, "crowd")
            if _alert_cooldown.get(cooldown_key, 0) <= now:
                detected, count, _ = process_crowd_detection(img)
                if detected:
                    _crowd_counter[cam_name] = _crowd_counter.get(cam_name, 0) + 1
                    if _crowd_counter[cam_name] >= 2:
                        # Set cooldown immediately to prevent duplicate processing
                        _alert_cooldown[cooldown_key] = now + 600
                        triggers["crowd"] = True
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        save_alert(timestamp, cam_name, "crowd", img, count, area_info)
                        _crowd_counter[cam_name] = 0
                else:
                    _crowd_counter[cam_name] = 0
            triggers["crowd"] = bool(detected)
            logging.info(f"[{cam_name}] Crowd Detection Status: {triggers['crowd']}")

        # Group 3: Loitering Detection
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

        # Group 4: Downtime Monitoring
        downtime_config = use_cases.get('downtime_monitoring', {})
        if downtime_config.get('enabled', False):
            logging.info(f"[{cam_name}] Checking: Downtime Monitoring")
            roi_configs = downtime_config.get('roi', [])
            frequency_sec = downtime_config.get('frequency_sec', 30)
            if roi_configs:
                # No cooldown needed - tracking handles timing internally
                detected, status, annotated_img, roi_name = process_downtime_monitoring(img, roi_configs, cam_name, frequency_sec,area_info=area_info)
                # Note: Alert is saved inside process_downtime_monitoring when transition from Red to Green occurs
                triggers["downtime_monitoring"] = bool(detected)
            logging.info(f"[{cam_name}] Downtime Monitoring Status: {triggers['downtime_monitoring']}")

        # Group 5: Camera Angle Check
        camera_angle_config = use_cases.get('camera_angle_check', {})
        if camera_angle_config.get('enabled', False):
            logging.info(f"[{cam_name}] Checking: Camera Angle Check")
            freq = camera_angle_config.get('frequency_sec', DEFAULT_FREQ_SEC)
            time_range = camera_angle_config.get('time', ['00:00', '23:59'])
            
            if time_range[0] <= current_time <= time_range[1]:
                angle_changed, annotated_img, deviation = process_camera_angle_check(
                    img, cam_name, min_matches=30, deviation_threshold=3.5
                )
                
                if angle_changed:
                    cooldown_key = (cam_name, 'camera_angle_check')
                    if cooldown_key not in _alert_cooldown or now >= _alert_cooldown[cooldown_key]:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        save_alert(timestamp, cam_name, "camera_angle_check", annotated_img, 1, area_info)
                        _alert_cooldown[cooldown_key] = now + (freq * 2)
                        triggers['camera_angle_check'] = True
            
            logging.info(f"[{cam_name}] Camera Angle Check Status: {triggers['camera_angle_check']} - {deviation}")

        # Group 6: Fire Exit Blocked Detection
        fire_exit_config = use_cases.get('fire_exit_blocked', {})
        if fire_exit_config.get('enabled', False):
            logging.info(f"[{cam_name}] Checking: Fire Exit Blocked Detection")
            detected = False
            cooldown_key = (cam_name, "fire_exit_blocked")
            if _alert_cooldown.get(cooldown_key, 0) <= now:
                # Set cooldown immediately to prevent duplicate processing
                _alert_cooldown[cooldown_key] = now + 300
                roi = fire_exit_config.get('roi', [])
                check_img = img.copy()
                # If ROI is given, crop to ROI polygon
                if roi and len(roi) > 0:
                    roi_polygon = roi[0] if isinstance(roi[0], list) else roi
                    roi_points = np.array(roi_polygon, dtype=np.int32)
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [roi_points], 255)
                    masked_img = cv2.bitwise_and(img, img, mask=mask)
                    x_coords = [p[0] for p in roi_points]
                    y_coords = [p[1] for p in roi_points]
                    x1, y1 = max(0, min(x_coords)), max(0, min(y_coords))
                    x2, y2 = min(img.shape[1], max(x_coords)), min(img.shape[0], max(y_coords))
                    check_img = masked_img[y1:y2, x1:x2]
                # If image is not clear, trigger alert
                if is_image_corrupted(check_img):
                    detected = False
                else:
                    # Use Gemini to check for non-essential items on ground
                    _, buffer = cv2.imencode(".jpg", check_img)
                    img_data = base64.b64encode(buffer).decode("utf-8")
                    prompt = (
                        "Fire Exit Blocked Detection:\n"
                        "Is there any non-essential item (like bags, boxes, obstacles) on the ground or blocking the fire exit?\n"
                        "If yes, reply 'yes'. If the area is clear, reply 'no'.\n"
                        "If the image is unclear or you are not sure, reply 'no'."
                    )
                    gemini_result = call_gemini_api(img_data, prompt)
                    if gemini_result == "yes":
                        detected = True
                if detected:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_alert(timestamp, cam_name, "fire_exit_blocked", img, 1, area_info)
            triggers["fire_exit_blocked"] = bool(detected)
            logging.info(f"[{cam_name}] Fire Exit Blocked Status: {triggers['fire_exit_blocked']}")

    except Exception as e:
        logging.error(f"Error processing camera {cam_name}: {e}")

    log_runtime(cam_name, triggers)

# ---------------------- SCHEDULER SETUP ----------------------
def schedule_device_tasks():
    """Set up periodic camera processing tasks from config file"""
    cameras_config = load_config()
    
    if not cameras_config:
        logging.error("No cameras found in configuration")
        return
    
    for cam_config in cameras_config:
        cam_name = cam_config.get("cam_name", "Unknown")
        use_cases = cam_config.get("use_cases", {})
        
        # Find minimum frequency among enabled use cases
        min_freq = None
        for v in use_cases.values():
            if v.get("enabled", False):
                freq = v.get("frequency_sec", DEFAULT_FREQ_SEC)
                if min_freq is None or freq < min_freq:
                    min_freq = freq
        
        if min_freq is not None:
            freq_sec = min_freq
            schedule.every(freq_sec).seconds.do(process_camera, cam_config)
            logging.info(f"Scheduled {cam_name} with frequency: {freq_sec}s")

# ---------------------- MAIN LOOP ----------------------
if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("WAAREE DEPLOYMENT SYSTEM STARTING...")
    logging.info("=" * 60)
    
    # Initialize ZMQ clients
    try:
        init_zmq_clients()
    except Exception as e:
        logging.error(f"Failed to initialize ZMQ clients: {e}")
        logging.error("Make sure person_detection_server.py and safety_detection_server.py are running")
        exit(1)
    
    schedule_device_tasks()
    
    # Print scheduled jobs
    logging.info("\nScheduled Jobs:")
    logging.info("-" * 60)
    for job in schedule.jobs:
        logging.info(f"Job: {job.job_func.func.__name__}")
        logging.info(f"Interval: Every {job.interval} {job.unit}")
        logging.info(f"Next run: {job.next_run}")
    logging.info("-" * 60)
    
    if not schedule.jobs:
        logging.error("No jobs scheduled. Check configuration file.")
        exit(1)
    
    logging.info("\nSystem running... Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\nShutting down gracefully...")
        # Cleanup ZMQ sockets
        if person_socket:
            person_socket.close()
        if safety_socket:
            safety_socket.close()
        if zmq_context:
            zmq_context.term()
        logging.info("System stopped")
