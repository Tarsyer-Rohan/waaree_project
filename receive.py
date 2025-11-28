from flask import Flask, request, jsonify
import os, json, time, threading
from datetime import datetime

# Configuration
BASE_FOLDER = "/home/waareeadmin/store_json/"  # storage for received files

# Alert types to skip (move to skipped_json folder)
SKIPPED_ALERT_TYPES = [

]

# Simplified folder structure - with separate retry folders
FOLDER_MAPPING = {
    "FOOTFALL": "footfall",     # All footfall data from all devices
    "ALERT": "alert",           # All alert data from all devices  
    "RETRY_FF": "retry_ff",     # Retry footfall files
    "RETRY_ALERTS": "retry_alerts",  # Retry alert files
    "SKIPPED": "skipped_json"   # Skipped alerts based on alert_type
}

os.makedirs(BASE_FOLDER, exist_ok=True)

app = Flask(__name__)

# Global variable to track if background sync is running
sync_running = False

def mongodb_sync_background():
    """Background function that runs every 5 minutes to sync files to MongoDB"""
    global sync_running
    sync_running = True
    
    while sync_running:
        try:
            print("🔄 Starting 5-minute MongoDB sync...")
            sync_files_to_mongodb()
            print("✅ 5-minute sync completed")
        except Exception as e:
            print(f"❌ Error in background sync: {e}")
        
        # Sleep for 5 minutes (300 seconds)
        for i in range(300):
            if not sync_running:
                break
            time.sleep(1)

def sync_files_to_mongodb():
    """Process accumulated JSON files and sync to MongoDB"""
    try:
        # This is a placeholder for the actual MongoDB sync logic
        # Count files to be processed
        footfall_folder = os.path.join(BASE_FOLDER, "footfall")
        alert_folder = os.path.join(BASE_FOLDER, "alert")
        retry_ff_folder = os.path.join(BASE_FOLDER, "retry_ff")
        retry_alerts_folder = os.path.join(BASE_FOLDER, "retry_alerts")
        skipped_folder = os.path.join(BASE_FOLDER, "skipped_json")
        
        files_to_process = 0
        if os.path.exists(footfall_folder):
            files_to_process += len([f for f in os.listdir(footfall_folder) if f.endswith('.json')])
        if os.path.exists(alert_folder):
            files_to_process += len([f for f in os.listdir(alert_folder) if f.endswith('.json')])
        if os.path.exists(retry_ff_folder):
            files_to_process += len([f for f in os.listdir(retry_ff_folder) if f.endswith('.json')])
        if os.path.exists(retry_alerts_folder):
            files_to_process += len([f for f in os.listdir(retry_alerts_folder) if f.endswith('.json')])
        if os.path.exists(skipped_folder):
            files_to_process += len([f for f in os.listdir(skipped_folder) if f.endswith('.json')])
        
        if files_to_process > 0:
            print(f"📊 Found {files_to_process} files to sync to MongoDB")
            # TODO: Add actual MongoDB sync logic here
            # This would call the enhanced MongoDB sync script
        else:
            print("📊 No new files to sync")
            
    except Exception as e:
        print(f"❌ MongoDB sync error: {e}")

def start_background_sync():
    """Start the background MongoDB sync thread"""
    sync_thread = threading.Thread(target=mongodb_sync_background, daemon=True)
    sync_thread.start()
    print("🚀 Background MongoDB sync started (5-minute intervals)")

def create_folder_structure(base_path, device_id, data_type=None):
    """Create simplified folder structure - only footfall and alert folders"""
    # Create only the main data type folder (no device-specific subfolders)
    if data_type:
        folder_name = FOLDER_MAPPING.get(data_type.upper(), "footfall")
        target_folder = os.path.join(base_path, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        print(f"📁 Using folder: {folder_name} for device {device_id}, type: {data_type}")
        return target_folder
    
    # Default to footfall folder if no data type specified
    default_folder = os.path.join(base_path, "footfall")
    os.makedirs(default_folder, exist_ok=True)
    return default_folder

def extract_data_info(data):
    """Extract device info and data type from payload"""
    # Get device ID (from serial_id or fallback)
    device_id = data.get("serial_id") or data.get("device_id", "unknown_device")
    
    # Get data type from the data
    raw_data_type = data.get("data_type", "ALERT")
    
    # Check if this alert should be skipped based on alert_type
    alert_type = data.get("alert_type")
    if alert_type and alert_type in SKIPPED_ALERT_TYPES:
        data_type = "SKIPPED"
        print(f"⏭️ Skipping alert type '{alert_type}' - moving to skipped_json folder")
    # Handle retry files - determine if it's retry footfall or retry alerts
    elif raw_data_type == "RETRY":
        # Check if this is originally a footfall or alert type retry
        # Look for indicators in the data to determine original type
        if (data.get("alert_type") or data.get("alert") or 
            any(key in data for key in ["crowd", "intrusion", "loitering","phn", 'ppe', ])):
            # This is a retry alert
            data_type = "RETRY_ALERTS"
        else:
            # This is a retry footfall
            data_type = "RETRY_FF"
    else:
        data_type = raw_data_type
    
    # Clean device ID (remove special characters for folder naming)
    device_id = "".join(c for c in str(device_id) if c.isalnum() or c in ['_', '-'])
    
    return device_id, data_type

def generate_filename(data, device_id, data_type):
    """Generate filename using format: {serial_id}_{camera_name}_{alert_type}_{timestamp}"""
    
    # Extract serial_id (prioritize serial_id from data, fallback to device_id)
    serial_id = data.get("serial_id", device_id)
    
    # Extract camera_name and clean it (remove special characters like spaces, $, %, etc.)
    camera_name = data.get("camera_no") or data.get("camera_name") or data.get("cam_name") or data.get("camera") or "unknown"
    # Replace spaces and special characters with underscores, keep alphanumeric and existing underscores/hyphens
    camera_name = "".join(c if (c.isalnum() or c in ['_', '-']) else '_' for c in str(camera_name))
    
    # For alert_type, try multiple possible fields based on data type
    if data_type.upper() == "SKIPPED":
        # For skipped alerts, use the actual alert_type
        alert_type = data.get("alert_type") or data.get("alert") or "skipped"
        print(f"Skipped alert type: {alert_type}")
    elif data_type.upper() in ["ALERT", "RETRY_ALERTS"]:
        # For alerts, look for alert_type, alert, type, or use data_type
        alert_type = data.get("alert_type") or data.get("alert") or data.get("type") or "alert"
        print(f"Alert type: {alert_type}")
    else:
        # For footfall/other types, use "ff" for footfall data
        alert_type = "ff"
    
    # Extract timestamp (prioritize date_time from JSON data)
    timestamp = None
    
    # Try different timestamp fields from the JSON data
    for time_field in ["date_time", "timestamp", "datetime", "time"]:
        if time_field in data:
            try:
                original_timestamp = data[time_field]
                
                # Handle different timestamp formats
                if "T" in str(original_timestamp):
                    # ISO format: 2025-11-11T14:30:25
                    dt = datetime.fromisoformat(str(original_timestamp).replace('Z', '+00:00'))
                    timestamp = dt.strftime("%Y%m%d_%H%M%S")
                elif "_" in str(original_timestamp):
                    # Format like: 2025-11-11_14-30-25
                    clean_timestamp = str(original_timestamp).replace(":", "").replace("-", "")
                    timestamp = clean_timestamp
                else:
                    # Try to parse other formats
                    timestamp = str(original_timestamp).replace(":", "").replace("-", "").replace(" ", "_")
                
                break  # Use the first valid timestamp found
                
            except Exception as e:
                print(f"⚠️ Error parsing timestamp field '{time_field}': {e}")
                continue
    
    # If no valid timestamp found in data, generate current timestamp (without milliseconds)
    if not timestamp:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Clean all parts (remove special characters, keep alphanumeric and underscores)
    serial_id = "".join(c for c in str(serial_id) if c.isalnum() or c in ['_', '-'])
    alert_type = "".join(c for c in str(alert_type) if c.isalnum() or c in ['_', '-'])
    timestamp = "".join(c for c in str(timestamp) if c.isalnum() or c in ['_', '-'])
    
    # Generate filename: {serial_id}_{camera_name}_{alert_type}_{timestamp}.json
    filename = f"{serial_id}_{camera_name}_{alert_type}_{timestamp}.json"
    
    return filename

@app.route("/upload", methods=["POST"])
def upload_json():
    """Receive single JSON from device"""
    try:
        # Check if request has JSON data
        if not request.is_json:
            print("❌ Request is not JSON")
            return jsonify({"status": "error", "message": "Request must be JSON"}), 400
            
        data = request.get_json(force=True)
        if not data:
            print("❌ Empty JSON data")
            return jsonify({"status": "error", "message": "Empty JSON data"}), 400
            
        print(f"📥 Received JSON data: {list(data.keys())}")  # Log the keys for debugging
        device_id, data_type = extract_data_info(data)
        
        result = store_single_file(data, device_id, data_type)
        return jsonify(result), 200 if result["status"] == "success" else 400
        
    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"status": "error", "message": error_msg}), 500

def store_single_file(data, device_id, data_type):
    """Store a single file and return result"""
    try:
        print(f"📥 Receiving from device: {device_id}, type: {data_type}")
        
        # Create file with proper folder structure
        target_folder = create_folder_structure(BASE_FOLDER, device_id, data_type)
        
        # Generate filename using the new format: {serial_id}_{alert_type}_{timestamp}
        filename = generate_filename(data, device_id, data_type)
        print(f"📝 Generated filename: {filename}")
        
        # Add server metadata with simplified format and converted timestamp
        current_time = datetime.utcnow()
        data['_server_metadata'] = {
            'received_timestamp': current_time.strftime("%Y-%m-%d %H-%M-%S"),
            'server_processed': True,
            'folder_type': FOLDER_MAPPING.get(data_type.upper(), "footfall")
        }
        
        # Add server_received_at in the desired format: YYYY-MM-DD HH-MM-SS
        data['server_received_at'] = current_time.strftime("%Y-%m-%d %H-%M-%S")
        filepath = os.path.join(target_folder, filename)
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        relative_path = os.path.relpath(filepath, BASE_FOLDER)
        print(f"✅ Stored: {relative_path}")
        
        return {
            "status": "success", 
            "file": filename,
            "device_id": device_id,
            "data_type": data_type,
            "path": relative_path
        }
        
    except Exception as e:
        print(f"❌ Error storing file for device {device_id}: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "device_id": device_id,
            "data_type": data_type
        }

@app.route("/status", methods=["GET"])
def get_status():
    """Get basic server status"""
    try:
        # Get simple folder stats for all folders
        total_files = 0
        footfall_files = 0
        alert_files = 0
        retry_ff_files = 0
        retry_alerts_files = 0
        skipped_files = 0
        
        # Count footfall files
        footfall_folder = os.path.join(BASE_FOLDER, "footfall")
        if os.path.exists(footfall_folder):
            footfall_files = len([f for f in os.listdir(footfall_folder) if f.endswith('.json')])
        
        # Count alert files
        alert_folder = os.path.join(BASE_FOLDER, "alert")
        if os.path.exists(alert_folder):
            alert_files = len([f for f in os.listdir(alert_folder) if f.endswith('.json')])
            
        # Count retry footfall files
        retry_ff_folder = os.path.join(BASE_FOLDER, "retry_ff")
        if os.path.exists(retry_ff_folder):
            retry_ff_files = len([f for f in os.listdir(retry_ff_folder) if f.endswith('.json')])
            
        # Count retry alert files
        retry_alerts_folder = os.path.join(BASE_FOLDER, "retry_alerts")
        if os.path.exists(retry_alerts_folder):
            retry_alerts_files = len([f for f in os.listdir(retry_alerts_folder) if f.endswith('.json')])
            
        # Count skipped files
        skipped_folder = os.path.join(BASE_FOLDER, "skipped_json")
        if os.path.exists(skipped_folder):
            skipped_files = len([f for f in os.listdir(skipped_folder) if f.endswith('.json')])
        
        total_files = footfall_files + alert_files + retry_ff_files + retry_alerts_files + skipped_files
        
        return jsonify({
            "status": "online",
            "total_files": total_files,
            "footfall_files": footfall_files,
            "alert_files": alert_files,
            "retry_ff_files": retry_ff_files,
            "retry_alerts_files": retry_alerts_files,
            "skipped_files": skipped_files,
            "skipped_alert_types": SKIPPED_ALERT_TYPES,
            "folder_structure": "footfall, alert, retry_ff, retry_alerts, and skipped_json"
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

if __name__ == "__main__":
    print("🚀 Starting simplified receiver server...")
    print(f"📂 Storage: {BASE_FOLDER}")
    print(f"📁 Structure: Five main folders - 'footfall', 'alert', 'retry_ff', 'retry_alerts', and 'skipped_json'")
    print(f"📋 All devices will store files in these folders based on data type")
    print(f"⏭️ Skipped alert types: {SKIPPED_ALERT_TYPES}")
    print(f"💡 To skip more alert types, add them to SKIPPED_ALERT_TYPES list")
    
    # Start the background MongoDB sync thread
    start_background_sync()
    
    app.run(host="0.0.0.0", port=5566)