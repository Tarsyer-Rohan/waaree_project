import os
import json
import glob
import logging
import time
import requests
import shutil
from datetime import datetime

# ================= CONFIGURATION =================
RECEIVER_URL = "http://192.168.105.4:5566/upload"   # Updated to match your receiver server
LOG_FILE_PATH = '/tmp/sending_local_data.log'

# =============== Logging Setup ===============
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO,
                    format='%(asctime)s: %(levelname)s: %(message)s')

# =============== Device ID ===============
serial_id = os.popen('sudo cat /sys/firmware/devicetree/base/serial-number').read().replace('\x00', '')
if not serial_id:
    serial_id = f"device_{int(time.time())}"

# =============== Folder Paths ===============
JSON_FOLDER_HOMEDIR_FF = '/home/pi/footfall_storage/'  # Retry footfall files
JSON_FOLDER_HOMEDIR_ALERTS = '/home/pi/alerts_storage/'  # Retry alert files (changed to match your request)
JSON_FOLDER_PATH = '/tmp/footfall_storage/'
ALERT_FOLDER_PATH = '/tmp/alerts/'

# Ensure folders exist  
for folder in [JSON_FOLDER_HOMEDIR_FF, JSON_FOLDER_HOMEDIR_ALERTS, JSON_FOLDER_PATH, ALERT_FOLDER_PATH]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        logging.info(f'Created folder: {folder}')

# =====================================================
def check_receiver_status():
    """Check if receiver server is online"""
    try:
        # Extract base URL from RECEIVER_URL
        base_url = RECEIVER_URL.replace("/upload", "")
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            logging.info(f"Receiver Status: {status_data.get('status', 'unknown')}")
            return True
        else:
            logging.warning(f"Receiver returned status code: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Cannot reach receiver: {e}")
        return False

def send_to_receiver(json_data, tag):
    """Send single JSON to receiver Flask server."""
    try:
        response = requests.post(RECEIVER_URL, json=json_data, timeout=10)
        
        # Parse response JSON for better error handling
        response_data = response.json()
        if response_data.get("status") == "success":
            logging.info(f"Receiver [{tag}]: ✅ Data sent successfully")
            return True
        else:
            logging.warning(f"Receiver [{tag}]: ⚠️ Server returned error: {response_data.get('message', 'Unknown error')}")
            return False

    except requests.Timeout:
        logging.warning(f"Receiver [{tag}]: ⚠️ Timeout occurred")
        return False
    except Exception as e:
        logging.error(f"Receiver [{tag}]: ❌ Request error {e}")
        return False

def process_json_file(json_file, tag):
    """Process a single JSON file: send → move to correct folder."""
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Add metadata and ensure receiver gets proper fields
        json_data["serial_id"] = json_data.get("serial_id", serial_id)
        json_data["data_type"] = tag
        json_data["device_sent_at"] = datetime.utcnow().isoformat()
        
        # Add date_time field that receiver prioritizes for filename
        if "date_time" not in json_data:
            # If no date_time in original data, create it from timestamp or current time
            if "timestamp" in json_data:
                json_data["date_time"] = json_data["timestamp"]
            else:
                json_data["date_time"] = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        result = send_to_receiver(json_data, tag)

        if result:
            # Successfully sent - delete the file completely
            os.remove(json_file)
            logging.info(f'[{tag}] ✅ File deleted after successful send: {json_file}')
        else:
            # Failed to send - move to appropriate retry folder based on data type
            if tag == "FOOTFALL" or tag == "RETRY_FF":
                retry_folder = JSON_FOLDER_HOMEDIR_FF
            elif tag == "ALERT" or tag == "RETRY_ALERTS":
                retry_folder = JSON_FOLDER_HOMEDIR_ALERTS
            else:
                # Default to footfall retry folder for unknown types
                retry_folder = JSON_FOLDER_HOMEDIR_FF
            
            # Use shutil.move to handle special characters in filenames properly
            filename = os.path.basename(json_file)
            destination = os.path.join(retry_folder, filename)
            shutil.move(json_file, destination)
            logging.warning(f'[{tag}] ⚠️ Failed to send; moved to retry folder: {destination}')

    except Exception as e:
        logging.error(f'[{tag}] ❌ Error processing file {json_file}: {e}')
        # Move to appropriate retry folder based on data type, even on error
        try:
            if tag == "FOOTFALL" or tag == "RETRY_FF":
                error_retry_folder = JSON_FOLDER_HOMEDIR_FF  # /home/pi/footfall_storage/
            elif tag == "ALERT" or tag == "RETRY_ALERTS":
                error_retry_folder = JSON_FOLDER_HOMEDIR_ALERTS  # /home/pi/alerts/
            else:
                # Default to footfall retry folder for unknown types
                error_retry_folder = JSON_FOLDER_HOMEDIR_FF
            
            filename = os.path.basename(json_file)
            error_destination = os.path.join(error_retry_folder, filename)
            shutil.move(json_file, error_destination)
            logging.info(f'[{tag}] Moved error file to retry folder: {error_destination}')
        except Exception as move_error:
            logging.error(f'[{tag}] ❌ Failed to move error file: {move_error}')
            # As last resort, try moving to /tmp/
            try:
                filename = os.path.basename(json_file)
                temp_destination = os.path.join('/tmp/', filename)
                shutil.move(json_file, temp_destination)
                logging.info(f'[{tag}] Moved error file to temp: {temp_destination}')
            except Exception as temp_error:
                logging.error(f'[{tag}] ❌ Failed to move error file to temp: {temp_error}')
    time.sleep(0.5)  # Reduced delay for batch processing


# =====================================================

# Log startup message only once
logging.info('=== DEVICE DATA SENDER STARTED ===')
logging.info(f"Serial ID: {serial_id}")

while True:
    la = [int(num) for num in os.getloadavg()]
    if all(x < 6 for x in la):
        
        # Check receiver status before sending
        if not check_receiver_status():
            logging.warning("⚠️ Receiver not available, skipping this round")
            time.sleep(60)  # Wait 1 minute before retry
            continue

        # 1️⃣ Retry footfall files from /home/pi/footfall_storage/
        retry_ff_files = glob.glob(f'{JSON_FOLDER_HOMEDIR_FF}/*.json')
        if retry_ff_files:
            logging.info(f"📁 Processing {len(retry_ff_files)} RETRY FOOTFALL files individually")
            for json_file in retry_ff_files:
                process_json_file(json_file, "RETRY_FF")
                time.sleep(0.1)  # Small delay between files

        # 1️⃣.5 Retry alert files from /home/pi/alert_storage/
        retry_alert_files = glob.glob(f'{JSON_FOLDER_HOMEDIR_ALERTS}/*.json')
        if retry_alert_files:
            logging.info(f"📁 Processing {len(retry_alert_files)} RETRY ALERT files individually")
            for json_file in retry_alert_files:
                process_json_file(json_file, "RETRY_ALERTS")
                time.sleep(0.1)  # Small delay between files

        # 2️⃣ Send footfall JSONs
        footfall_files = glob.glob(f'{JSON_FOLDER_PATH}/*.json')
        if footfall_files:
            logging.info(f"📁 Processing {len(footfall_files)} FOOTFALL files individually")
            for json_file in footfall_files:
                process_json_file(json_file, "FOOTFALL")
                time.sleep(0.1)  # Small delay between files

        # 3️⃣ Send alert JSONs
        alert_files = glob.glob(f'{ALERT_FOLDER_PATH}/*.json')
        if alert_files:
            logging.info(f"📁 Processing {len(alert_files)} ALERT files individually")
            for json_file in alert_files:
                process_json_file(json_file, "ALERT")
                time.sleep(0.1)  # Small delay between files

    else:
        logging.warning(f'⚠️ High system load: {la}, skipping this round')

    # Wait before next iteration
    time.sleep(180)