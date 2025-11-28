import os, json, hashlib, base64, time
from datetime import datetime
from pymongo import MongoClient, ASCENDING

# -------- Configuration --------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME", "device_dashboard")
FOOT_DIR  = os.getenv("FOOT_DIR", "/home/waareeadmin/store_json/footfall")
ALERT_DIR = os.getenv("ALERT_DIR", "/home/waareeadmin/store_json/alert")
DONE_DIR  = os.getenv("DONE_DIR", "/home/waareeadmin/store_json/deleted_json")
ALERT_IMG_DIR = "/mnt/hdd/alert_images"   # Folder to store decoded images
# --------------------------------

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
foot = db["footfall"]
alrt = db["alerts"]

# Ensure indexes exist
for coll in (foot, alrt):
    coll.create_index([("server_received_at", ASCENDING)])
    coll.create_index([("serial_id", ASCENDING)])  # Use serial_id instead of device_id
    coll.create_index([("doc_hash", ASCENDING)], unique=True)
    coll.create_index([("area", ASCENDING)])  # Index for area field

def sha(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def ensure_list(x): 
    return x if isinstance(x, list) else [x]

def save_alert_image(doc):
    try:
        if not os.path.isdir(ALERT_IMG_DIR):
            os.makedirs(ALERT_IMG_DIR, exist_ok=True)

        alert_type = doc.get("alert_type", "unknown").replace(" ", "_")
        camera_no = doc.get("camera_no", "unknown").replace(" ", "_").replace("/", "_")
        date_time = doc.get("date_time", datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")).replace(":", "-").replace(" ", "_")

        filename = f"{alert_type}_{camera_no}_{date_time}.jpg"
        save_path = os.path.join(ALERT_IMG_DIR, filename)

        if "image_byte_str" in doc and isinstance(doc["image_byte_str"], str):
            b64data = doc["image_byte_str"].strip()
            if len(b64data) > 100:
                try:
                    with open(save_path, "wb") as img_file:
                        img_file.write(base64.b64decode(b64data))
                    # Save only the URL path for Flask compatibility
                    doc["image_byte_str"] = f"/alert_images/{filename}"
                    print(f"[ALERT_IMG] Saved image: {save_path}")
                except Exception as img_err:
                    print(f"[ALERT_IMG] Failed to save image: {img_err}")
            else:
                print("[ALERT_IMG] Skipped — not valid base64 content.")
        else:
            print("[ALERT_IMG] No image data found in alert.")
    except Exception as e:
        print(f"[ALERT_IMG] Error: {e}")

def normalize(doc, dtype, src):
    d = dict(doc)
    d["data_type"] = dtype
    d["source_file"] = os.path.basename(src)
    # add TID = filename without .json
    d["tid"] = os.path.splitext(os.path.basename(src))[0]
    
    # Remove unwanted fields: timestamp and device_id
    d.pop("timestamp", None)
    d.pop("device_id", None)
    
    # Ensure area field is preserved if it exists
    if "area" in doc:
        d["area"] = doc["area"]
    
    # Use the same format as receive.py: YYYY-MM-DD HH-MM-SS
    # Only set server_received_at if it's not already in the document with the correct format
    if "server_received_at" not in d or "T" in str(d.get("server_received_at", "")):
        d["server_received_at"] = datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S")
    
    return d

def handle(folder, coll, dtype):
    if not os.path.isdir(folder):
        return
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            docs = [normalize(x, dtype, path) for x in ensure_list(data)]
            for d in docs:
                # Only insert alert-type data into alerts collection
                if coll.name == "alerts":
                    if d.get("data_type", "ALERT").upper() != "ALERT":
                        continue
                if dtype == "ALERT" and "image_byte_str" in d:
                    save_alert_image(d)
                d["doc_hash"] = sha(d)
                coll.update_one({"doc_hash": d["doc_hash"]}, {"$setOnInsert": d}, upsert=True)
            os.makedirs(DONE_DIR, exist_ok=True)
            os.replace(path, os.path.join(DONE_DIR, name))
            print(f"[{dtype}] Inserted {path}")
        except Exception as e:
            print(f"[{dtype}] Error {path}: {e}", flush=True)

def main():
    handle(FOOT_DIR, foot, "FOOTFALL")
    handle(ALERT_DIR, alrt, "ALERT")

if __name__ == "__main__":
    print("🚀 Starting MongoDB sync service (1-minute intervals)...")
    while True:
        try:
            print(f"\n🔄 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting sync cycle...")
            main()
            print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sync completed")
        except Exception as e:
            print(f"❌ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error during sync: {e}")
        
        # Wait for 1 minute (60 seconds)
        print(f"⏳ Waiting 60 seconds until next sync...")
        time.sleep(60)