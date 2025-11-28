from flask import Flask, render_template, jsonify, request, send_from_directory
from pymongo import MongoClient
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017")
db = client["device_dashboard"]
alerts_col = db["alerts"]
footfall_col = db["footfall"]

# Helper function to process image URLs
def process_image_url(image_path):
    """Remove /mnt/hdd prefix from image paths."""
    if not image_path:
        return image_path
    # Remove /mnt/hdd prefix if present
    if image_path.startswith('/mnt/hdd/alert_images/'):
        return '/alert_images/' + image_path.split('/alert_images/')[-1]
    elif image_path.startswith('/mnt/hdd/'):
        return image_path.replace('/mnt/hdd/', '/')
    return image_path


# ---------- ROUTES ---------- #
@app.route("/footfall_vehicle")
def footfall_vehicle_page():
    return render_template("footfall_vehicle.html")

@app.route("/footfall")
def footfall_page():
    return render_template("footfall.html")

@app.route("/vehicle_exit")
def vehicle_exit_page():
    return render_template("vehicle_exit.html")
# ---------- API ROUTES ---------- #

# --- Footfall & Vehicle Exit APIs --- #

def build_ff_vf_query(request_args, type_key):
    start_date = request_args.get("start_date")
    end_date = request_args.get("end_date")
    cam_no = request_args.get("cam_no")
    query = {}
    if start_date and end_date:
        query["date_time"] = {
            "$gte": start_date + " 00:00:00",
            "$lte": end_date + " 23:59:59"
        }
    # Filter by FF or VF in store_code
    if cam_no and cam_no != "All Cameras":
        # If camera number is specified, use exact match pattern
        query["store_code"] = {"$regex": f"-{type_key}-{cam_no}$"}
    else:
        # Otherwise just filter by type
        if type_key == "FF":
            query["store_code"] = {"$regex": "-FF-"}
        elif type_key == "VF":
            query["store_code"] = {"$regex": "-VF-"}
    return query

@app.route("/api/footfall_cameras")
def api_footfall_cameras():
    """Return unique camera numbers from footfall collection for FF."""
    try:
        store_codes = footfall_col.distinct("store_code", {"store_code": {"$regex": "-FF-"}})
        cameras = set()
        for code in store_codes:
            # Extract camera number from format: waaree-FF-{camno}
            if "-FF-" in code:
                try:
                    cam_no = code.split("-FF-")[-1]
                    if cam_no:
                        cameras.add(cam_no)
                except Exception:
                    continue
        return jsonify({"ok": True, "cameras": sorted(list(cameras))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/vehicle_exit_cameras")
def api_vehicle_exit_cameras():
    """Return unique camera numbers from footfall collection for VF."""
    try:
        store_codes = footfall_col.distinct("store_code", {"store_code": {"$regex": "-VF-"}})
        cameras = set()
        for code in store_codes:
            # Extract camera number from format: waaree-VF-{camno}
            if "-VF-" in code:
                try:
                    cam_no = code.split("-VF-")[-1]
                    if cam_no:
                        cameras.add(cam_no)
                except Exception:
                    continue
        return jsonify({"ok": True, "cameras": sorted(list(cameras))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/footfall_daily")
def api_footfall_daily():
    query = build_ff_vf_query(request.args, "FF")
    cursor = footfall_col.find(query, {"date_time": 1, "count_child": 1, "count_female": 1, "count_male": 1, "count_staff": 1, "store_code": 1, "_id": 0})
    data = list(cursor)
    daily_counts = {}
    for doc in data:
        dt_str = doc.get("date_time", "")
        try:
            date_str = dt_str.split(" ")[0] if " " in dt_str else dt_str[:10]
        except Exception:
            continue
        if date_str:
            daily_counts.setdefault(date_str, 0)
            daily_counts[date_str] += int(doc.get("count_child", 0)) + int(doc.get("count_female", 0)) + int(doc.get("count_male", 0)) + int(doc.get("count_staff", 0))
    dates = sorted(daily_counts.keys())
    counts = [daily_counts[d] for d in dates]
    return jsonify({"ok": True, "dates": dates, "counts": counts})

@app.route("/api/footfall_hourly")
def api_footfall_hourly():
    query = build_ff_vf_query(request.args, "FF")
    cursor = footfall_col.find(query, {"date_time": 1, "count_child": 1, "count_female": 1, "count_male": 1, "count_staff": 1, "store_code": 1, "_id": 0})
    data = list(cursor)
    hourly_counts = {h: 0 for h in range(24)}
    for doc in data:
        dt_str = doc.get("date_time", "")
        hour = None
        try:
            if " " in dt_str:
                time_part = dt_str.split(" ")[1]
                if ":" in time_part:
                    hour = int(time_part.split(":")[0])
                elif "-" in time_part:
                    hour = int(time_part.split("-")[0])
        except Exception:
            continue
        if hour is not None and 0 <= hour <= 23:
            hourly_counts[hour] += int(doc.get("count_child", 0)) + int(doc.get("count_female", 0)) + int(doc.get("count_male", 0)) + int(doc.get("count_staff", 0))
    hours = list(range(24))
    counts = [hourly_counts[h] for h in hours]
    return jsonify({"ok": True, "hours": hours, "counts": counts})

@app.route("/api/vehicle_exit_daily")
def api_vehicle_exit_daily():
    query = build_ff_vf_query(request.args, "VF")
    cursor = footfall_col.find(query, {"date_time": 1, "count_child": 1, "count_female": 1, "count_male": 1, "count_staff": 1, "store_code": 1, "_id": 0})
    data = list(cursor)
    daily_counts = {}
    for doc in data:
        dt_str = doc.get("date_time", "")
        try:
            date_str = dt_str.split(" ")[0] if " " in dt_str else dt_str[:10]
        except Exception:
            continue
        if date_str:
            daily_counts.setdefault(date_str, 0)
            daily_counts[date_str] += int(doc.get("count_child", 0)) + int(doc.get("count_female", 0)) + int(doc.get("count_male", 0)) + int(doc.get("count_staff", 0))
    dates = sorted(daily_counts.keys())
    counts = [daily_counts[d] for d in dates]
    return jsonify({"ok": True, "dates": dates, "counts": counts})

@app.route("/api/vehicle_exit_hourly")
def api_vehicle_exit_hourly():
    query = build_ff_vf_query(request.args, "VF")
    cursor = footfall_col.find(query, {"date_time": 1, "count_child": 1, "count_female": 1, "count_male": 1, "count_staff": 1, "store_code": 1, "_id": 0})
    data = list(cursor)
    hourly_counts = {h: 0 for h in range(24)}
    for doc in data:
        dt_str = doc.get("date_time", "")
        hour = None
        try:
            if " " in dt_str:
                time_part = dt_str.split(" ")[1]
                if ":" in time_part:
                    hour = int(time_part.split(":")[0])
                elif "-" in time_part:
                    hour = int(time_part.split("-")[0])
        except Exception:
            continue
        if hour is not None and 0 <= hour <= 23:
            hourly_counts[hour] += int(doc.get("count_child", 0)) + int(doc.get("count_female", 0)) + int(doc.get("count_male", 0)) + int(doc.get("count_staff", 0))
    hours = list(range(24))
    counts = [hourly_counts[h] for h in hours]
    return jsonify({"ok": True, "hours": hours, "counts": counts})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")

@app.route("/production")
def production_page():
    return render_template("production.html")

@app.route("/compliance")
def compliance_page():
    return render_template("compliance.html")

@app.route("/production_alerts")
def production_alerts_page():
    return render_template("production_alerts.html")

@app.route("/alert_images/<path:filename>")
def serve_alert_image(filename):
    """Serve alert images from /mnt/hdd/alert_images/ directory."""
    image_dir = "/mnt/hdd/alert_images"
    return send_from_directory(image_dir, filename)

# ---------- API ROUTES ---------- #

@app.route("/api/alerts_data")
def alerts_data():
    """Return filtered alert data for table with proper filtering."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    area = request.args.get("area")
    line = request.args.get("line")
    shift = request.args.get("shift")
    alert_type = request.args.get("alert_type")
    camera = request.args.get("camera")
    section = request.args.get("section", "strategic")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    query = {}
    if start_date and end_date:
        query["date_time"] = {
            "$gte": start_date + " 00:00:00",
            "$lte": end_date + " 23:59:59"
        }
    
    # Handle filtering based on section
    if section == "compliance":
        # Compliance section: Handle EHS and FIRE areas
        if area and area != "All Areas":
            if area == "EHS":
                # EHS: All alert types except fire_exit_blocked
                query["alert_type"] = {"$ne": "fire_exit_blocked"}
            elif area == "FIRE":
                # FIRE: Only fire_exit_blocked
                query["alert_type"] = "fire_exit_blocked"
        # If "All Areas", show all compliance data (no additional filter needed)
        
        # Zone filter for compliance (e.g., MLB-1, CLB-2, etc.) - overrides area logic
        zone = request.args.get("zone")
        if zone and zone != "All Zones":
            # Match zone and all its sub-zones (e.g., MLB-2 matches MLB-2, MLB-2-LINE-10, etc.)
            query["area"] = {"$regex": f"^{zone}"}
    else:
        # Strategic area: Exclude production areas (all MLB and CLB)
        query["area"] = {"$not": {"$regex": "^(MLB|CLB)"}}
        
        if area and area != "All Areas":
            # Override area filter if specific area selected - use case-insensitive regex
            query["area"] = {"$regex": area, "$options": "i"}
    
    if line and line != "All Lines":
        query["camera_no"] = {"$regex": f"Line-{line}"}
    if shift and shift != "All Shifts":
        query["shift"] = shift
    
    # Camera filter
    if camera and camera != "All Cameras":
        query["camera_no"] = camera
    
    # Alert type filter using contains logic
    if alert_type and alert_type != "All Alert Types":
        query["alert_type"] = {"$regex": alert_type, "$options": "i"}

    # Get total count
    total_count = alerts_col.count_documents(query)
    
    # Calculate skip value for pagination
    skip = (page - 1) * per_page
    
    # Get paginated data
    data = list(alerts_col.find(query, {"_id": 0}).sort("date_time", -1).skip(skip).limit(per_page))
    
    # Process image URLs to remove /mnt/hdd prefix
    for item in data:
        if 'image_byte_str' in item:
            item['image_byte_str'] = process_image_url(item['image_byte_str'])
    
    return jsonify({
        "ok": True, 
        "data": data,
        "total": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": (total_count + per_page - 1) // per_page
    })

@app.route("/api/alerts_lines")
def alerts_lines():
    """Return unique line numbers extracted from camera_no field (excluding MLB/CLB)."""
    try:
        section = request.args.get("section", "strategic")
        
        # Build query based on section
        if section == "compliance":
            # For compliance, get all lines (no area filter)
            query = {}
        else:
            # For strategic, exclude production areas
            query = {"area": {"$not": {"$regex": "^(MLB|CLB)"}}}
        
        camera_nos = alerts_col.distinct("camera_no", query)
        lines = set()
        
        for camera_no in camera_nos:
            if camera_no and "Line-" in camera_no:
                try:
                    # Extract line number from formats like "M-2_L-12_VQC_(ALT)-Line-12"
                    line_part = camera_no.split("Line-")[-1]
                    # Extract just the number (in case there's more text after)
                    line_num = ""
                    for char in line_part:
                        if char.isdigit():
                            line_num += char
                        else:
                            break
                    if line_num:
                        lines.add(line_num)
                except Exception:
                    continue
        
        return jsonify({"ok": True, "lines": sorted(list(lines), key=lambda x: int(x))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/alerts_time_series")
def alerts_time_series():
    """Return daily trend grouped by alert_type with proper filtering."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    area = request.args.get("area")
    line = request.args.get("line")
    shift = request.args.get("shift")
    alert_type = request.args.get("alert_type")
    camera = request.args.get("camera")
    section = request.args.get("section", "strategic")

    query = {}
    if start_date and end_date:
        query["date_time"] = {
            "$gte": start_date + " 00:00:00",
            "$lte": end_date + " 23:59:59"
        }
    
    # Handle filtering based on section
    if section == "compliance":
        # Compliance section: Handle EHS and FIRE areas
        if area and area != "All Areas":
            if area == "EHS":
                # EHS: All alert types except fire_exit_blocked
                query["alert_type"] = {"$ne": "fire_exit_blocked"}
            elif area == "FIRE":
                # FIRE: Only fire_exit_blocked
                query["alert_type"] = "fire_exit_blocked"
    else:
        # Strategic area: Exclude production areas (all MLB and CLB)
        query["area"] = {"$not": {"$regex": "^(MLB|CLB)"}}
        
        if area and area != "All Areas":
            # Override area filter if specific area selected - use case-insensitive regex
            query["area"] = {"$regex": area, "$options": "i"}
    
    if line and line != "All Lines":
        query["camera_no"] = {"$regex": f"Line-{line}"}
    if shift and shift != "All Shifts":
        query["shift"] = shift
    
    # Camera filter
    if camera and camera != "All Cameras":
        query["camera_no"] = camera
    
    # Alert type filter using contains logic
    if alert_type and alert_type != "All Alert Types":
        query["alert_type"] = {"$regex": alert_type, "$options": "i"}

    cursor = alerts_col.find(query, {"alert_type": 1, "date_time": 1, "area": 1, "_id": 0})
    data = list(cursor)
    
    daily_counts = {}
    types = set()

    for doc in data:
        alert_type = doc.get("alert_type", "Unknown")
        area = doc.get("area", "Unknown")
        dt_str = doc.get("date_time", "")
        try:
            # Extract date from "2025-11-05 06:22:37" format
            date_str = dt_str.split(" ")[0] if " " in dt_str else dt_str[:10]
        except Exception:
            continue
            
        if date_str:
            # For compliance, group by EHS/FIRE areas
            if section == "compliance":
                if alert_type == "fire_exit_blocked":
                    area_label = "Fire and Emergency Readiness (ISO 14001)"
                else:
                    area_label = "EHS Safety Compliance (OSHA Factory Act, ISO 45001)"
                daily_counts.setdefault(date_str, {})
                daily_counts[date_str].setdefault(area_label, 0)
                daily_counts[date_str][area_label] += 1
                types.add(area_label)
            else:
                # For strategic, group by area (periphery, Road area, etc.)
                daily_counts.setdefault(date_str, {})
                daily_counts[date_str].setdefault(area, 0)
                daily_counts[date_str][area] += 1
                types.add(area)

    # Get sorted list of dates from the data
    dates = sorted(daily_counts.keys()) if daily_counts else []
    series = {t: [daily_counts.get(d, {}).get(t, 0) for d in dates] for t in types}

    return jsonify({"ok": True, "dates": dates, "types": sorted(list(types)), "series": series})

@app.route("/api/alerts_hourly")
def alerts_hourly():
    """Return hour-wise stacked counts for all 24 hours with proper filtering."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    area = request.args.get("area")
    line = request.args.get("line")
    shift = request.args.get("shift")
    alert_type = request.args.get("alert_type")
    camera = request.args.get("camera")
    section = request.args.get("section", "strategic")
    
    print(f"Hourly API called with: section={section}, area={area}, start={start_date}, end={end_date}")

    query = {}
    if start_date and end_date:
        query["date_time"] = {
            "$gte": start_date + " 00:00:00",
            "$lte": end_date + " 23:59:59"
        }
    
    # Handle filtering based on section
    if section == "compliance":
        # Compliance section: Handle EHS and FIRE areas
        if area and area != "All Areas":
            if area == "EHS":
                # EHS: All alert types except fire_exit_blocked
                query["alert_type"] = {"$ne": "fire_exit_blocked"}
            elif area == "FIRE":
                # FIRE: Only fire_exit_blocked
                query["alert_type"] = "fire_exit_blocked"
    else:
        # Strategic area: Exclude production areas (all MLB and CLB)
        query["area"] = {"$not": {"$regex": "^(MLB|CLB)"}}
        
        if area and area != "All Areas":
            # Override area filter if specific area selected - use case-insensitive regex
            query["area"] = {"$regex": area, "$options": "i"}
    
    if line and line != "All Lines":
        query["camera_no"] = {"$regex": f"Line-{line}"}
    if shift and shift != "All Shifts":
        query["shift"] = shift
    
    # Camera filter
    if camera and camera != "All Cameras":
        query["camera_no"] = camera
    
    # Alert type filter using contains logic
    if alert_type and alert_type != "All Alert Types":
        query["alert_type"] = {"$regex": alert_type, "$options": "i"}
    
    # Alert type filter using contains logic
    if alert_type and alert_type != "All Alert Types":
        query["alert_type"] = {"$regex": alert_type, "$options": "i"}

    cursor = alerts_col.find(query, {"alert_type": 1, "date_time": 1, "area": 1, "_id": 0})
    data = list(cursor)
    
    print(f"Hourly query: {query}")
    print(f"Hourly data count: {len(data)}")
    if len(data) > 0:
        print(f"Sample hourly doc: {data[0]}")
    
    hour_counts = {}
    types = set()

    for doc in data:
        alert_type = doc.get("alert_type", "Unknown")
        area = doc.get("area", "Unknown")
        dt_str = doc.get("date_time", "")
        try:
            # Support both "2025-11-05 06:22:37" and "2025-11-12 14-26-18" formats
            if " " in dt_str:
                time_part = dt_str.split(" ")[1]
                if ":" in time_part:
                    hour = int(time_part.split(":")[0])
                elif "-" in time_part:
                    hour = int(time_part.split("-")[0])
                else:
                    continue
            else:
                continue
        except Exception:
            continue
        
        if 0 <= hour <= 23:
            # For compliance, group by EHS/FIRE areas
            if section == "compliance":
                if alert_type == "fire_exit_blocked":
                    area_label = "Fire and Emergency Readiness (ISO 14001)"
                else:
                    area_label = "EHS Safety Compliance (OSHA Factory Act, ISO 45001)"
                hour_counts.setdefault(hour, {})
                hour_counts[hour].setdefault(area_label, 0)
                hour_counts[hour][area_label] += 1
                types.add(area_label)
            else:
                # For strategic, group by area (periphery, Road area, etc.)
                hour_counts.setdefault(hour, {})
                hour_counts[hour].setdefault(area, 0)
                hour_counts[hour][area] += 1
                types.add(area)

    hours = list(range(0, 24))
    series = {t: [hour_counts.get(h, {}).get(t, 0) for h in hours] for t in types}
    
    print(f"Hourly types: {types}")
    print(f"Hourly series keys: {series.keys()}")
    print(f"Hourly hour_counts sample: {dict(list(hour_counts.items())[:3]) if hour_counts else {}}")

    return jsonify({"ok": True, "hours": hours, "types": sorted(list(types)), "series": series})

# ---------- PRODUCTION API ROUTES ---------- #

@app.route("/api/production_data")
def production_data():
    """Return filtered production alert data (MLB & CLB areas)."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    area = request.args.get("area")
    line = request.args.get("line")
    shift = request.args.get("shift")
    alert_type = request.args.get("alert_type")
    camera = request.args.get("camera")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    
    print(f"Production API called with params: start_date={start_date}, end_date={end_date}, area={area}, line={line}, shift={shift}, alert_type={alert_type}, camera={camera}, page={page}")

    query = {}
    if start_date and end_date:
        query["date_time"] = {
            "$gte": start_date + " 00:00:00",
            "$lte": end_date + " 23:59:59"
        }
    
    # Handle area mapping for production
    if area and area != "All Areas":
        if area == "Module Manufacturing":
            query["area"] = {"$regex": "^MLB"}
        elif area == "Cell Manufacturing":
            query["area"] = {"$regex": "^CLB"}
    else:
        # By default, show all MLB and CLB areas for production
        query["area"] = {"$regex": "^(MLB|CLB)"}
    
    # Zone filter (e.g., MLB-1, CLB-2, etc.) - overrides area filter
    zone = request.args.get("zone")
    if zone and zone != "All Zones":
        # Match zone and all its sub-zones (e.g., MLB-2 matches MLB-2, MLB-2-LINE-10, etc.)
        query["area"] = {"$regex": f"^{zone}"}
    
    if line and line != "All Lines":
        query["camera_no"] = {"$regex": f"Line-{line}"}
    if shift and shift != "All Shifts":
        query["shift"] = shift
    
    # Camera filter
    if camera and camera != "All Cameras":
        query["camera_no"] = camera
    
    # Alert type filter using contains logic
    if alert_type and alert_type != "All Alert Types":
        query["alert_type"] = {"$regex": alert_type, "$options": "i"}

    print(f"Production query: {query}")
    
    try:
        # Get total count
        total_count = alerts_col.count_documents(query)
        
        # Calculate skip value for pagination
        skip = (page - 1) * per_page
        
        # Get paginated data
        data = list(alerts_col.find(query, {"_id": 0}).sort("date_time", -1).skip(skip).limit(per_page))
        print(f"Found {len(data)} production records out of {total_count} total")
        if len(data) > 0:
            print(f"Data Present")
        
        # Process image URLs to remove /mnt/hdd prefix
        for item in data:
            if 'image_byte_str' in item:
                item['image_byte_str'] = process_image_url(item['image_byte_str'])
        
        return jsonify({
            "ok": True, 
            "data": data,
            "total": total_count,
            "page": page,
            "per_page": per_page,
            "total_pages": (total_count + per_page - 1) // per_page
        })
    except Exception as e:
        print(f"Error in production_data: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/production_lines")
def production_lines():
    """Return unique line numbers from production areas (MLB & CLB)."""
    try:
        camera_nos = alerts_col.distinct("camera_no", {"area": {"$regex": "^(MLB|CLB)"}})
        lines = set()
        
        for camera_no in camera_nos:
            if camera_no and "Line-" in camera_no:
                try:
                    line_part = camera_no.split("Line-")[-1]
                    line_num = ""
                    for char in line_part:
                        if char.isdigit():
                            line_num += char
                        else:
                            break
                    if line_num:
                        lines.add(line_num)
                except Exception:
                    continue
        
        return jsonify({"ok": True, "lines": sorted(list(lines), key=lambda x: int(x))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/compliance_zones")
def compliance_zones():
    """Return unique zone names from compliance areas (MLB-1, MLB-2, CLB-1, CLB-2, etc.)."""
    try:
        areas = alerts_col.distinct("area", {"area": {"$regex": "^(MLB|CLB)"}})
        # Extract base zone names without line numbers (e.g., MLB-2, CLB-1)
        zones = set()
        for area in areas:
            if area:
                # Match pattern like MLB-1, MLB-2, CLB-1, etc. (ignore line suffixes)
                import re
                match = re.match(r'^(MLB-\d+|CLB-\d+)', area)
                if match:
                    zones.add(match.group(1))
        return jsonify({"ok": True, "zones": sorted(list(zones))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/compliance_cameras")
def compliance_cameras():
    """Return unique camera numbers from compliance areas."""
    try:
        cameras = alerts_col.distinct("camera_no", {"area": {"$regex": "^(MLB|CLB)"}})
        camera_list = sorted([cam for cam in cameras if cam])
        return jsonify({"ok": True, "cameras": camera_list})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/strategic_cameras")
def strategic_cameras():
    """Return unique camera numbers from strategic areas (non-MLB/CLB)."""
    try:
        cameras = alerts_col.distinct("camera_no", {"area": {"$not": {"$regex": "^(MLB|CLB)"}}})
        camera_list = sorted([cam for cam in cameras if cam])
        return jsonify({"ok": True, "cameras": camera_list})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/production_cameras")
def production_cameras():
    """Return unique camera numbers from production areas."""
    try:
        cameras = alerts_col.distinct("camera_no", {"area": {"$regex": "^(MLB|CLB)"}})
        camera_list = sorted([cam for cam in cameras if cam])
        return jsonify({"ok": True, "cameras": camera_list})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/production_time_series")
def production_time_series():
    """Return daily trend for production areas (MLB-2 & CLB-1) grouped by alert_type."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    area = request.args.get("area")
    line = request.args.get("line")
    shift = request.args.get("shift")
    alert_type = request.args.get("alert_type")
    camera = request.args.get("camera")

    query = {}
    if start_date and end_date:
        query["date_time"] = {
            "$gte": start_date + " 00:00:00",
            "$lte": end_date + " 23:59:59"
        }
    
    # Handle area mapping for production
    if area and area != "All Areas":
        if area == "Module Manufacturing":
            query["area"] = {"$regex": "^MLB"}
        elif area == "Cell Manufacturing":
            query["area"] = {"$regex": "^CLB"}
    else:
        # By default, show all MLB and CLB areas for production
        query["area"] = {"$regex": "^(MLB|CLB)"}
    
    # Zone filter (e.g., MLB-1, CLB-2, etc.) - overrides area filter
    zone = request.args.get("zone")
    if zone and zone != "All Zones":
        # Match zone and all its sub-zones (e.g., MLB-2 matches MLB-2, MLB-2-LINE-10, etc.)
        query["area"] = {"$regex": f"^{zone}"}
    
    if line and line != "All Lines":
        query["camera_no"] = {"$regex": f"Line-{line}"}
    if shift and shift != "All Shifts":
        query["shift"] = shift
    
    # Camera filter
    if camera and camera != "All Cameras":
        query["camera_no"] = camera
    
    # Alert type filter using contains logic
    if alert_type and alert_type != "All Alert Types":
        query["alert_type"] = {"$regex": alert_type, "$options": "i"}

    cursor = alerts_col.find(query, {"alert_type": 1, "date_time": 1, "area": 1, "_id": 0})
    data = list(cursor)
    
    daily_counts = {}
    types = set()

    for doc in data:
        area = doc.get("area", "")
        dt_str = doc.get("date_time", "")
        try:
            date_str = dt_str.split(" ")[0] if " " in dt_str else dt_str[:10]
        except Exception:
            continue
            
        if date_str:
            # Determine area based on area
            if area.startswith("MLB"):
                area_label = "Module Manufacturing"
            elif area.startswith("CLB"):
                area_label = "Cell Manufacturing"
            else:
                area_label = area
            
            daily_counts.setdefault(date_str, {})
            daily_counts[date_str].setdefault(area_label, 0)
            daily_counts[date_str][area_label] += 1
            types.add(area_label)

    dates = sorted(daily_counts.keys()) if daily_counts else []
    series = {t: [daily_counts.get(d, {}).get(t, 0) for d in dates] for t in types}

    return jsonify({"ok": True, "dates": dates, "types": sorted(list(types)), "series": series})

@app.route("/api/production_hourly")
def production_hourly():
    """Return hour-wise stacked counts for all 24 hours for production areas (MLB & CLB)."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    area = request.args.get("area")
    camera = request.args.get("camera")
    line = request.args.get("line")
    shift = request.args.get("shift")
    alert_type = request.args.get("alert_type")

    query = {}
    if start_date and end_date:
        query["date_time"] = {
            "$gte": start_date + " 00:00:00",
            "$lte": end_date + " 23:59:59"
        }
    
    # Handle area mapping for production
    if area and area != "All Areas":
        if area == "Module Manufacturing":
            query["area"] = {"$regex": "^MLB"}
        elif area == "Cell Manufacturing":
            query["area"] = {"$regex": "^CLB"}
    else:
        # By default, show all MLB and CLB areas for production
        query["area"] = {"$regex": "^(MLB|CLB)"}
    
    # Zone filter (e.g., MLB-1, CLB-2, etc.) - overrides area filter
    zone = request.args.get("zone")
    if zone and zone != "All Zones":
        # Match zone and all its sub-zones (e.g., MLB-2 matches MLB-2, MLB-2-LINE-10, etc.)
        query["area"] = {"$regex": f"^{zone}"}
    
    if line and line != "All Lines":
        query["camera_no"] = {"$regex": f"Line-{line}"}
    if shift and shift != "All Shifts":
        query["shift"] = shift
    
    # Camera filter
    if camera and camera != "All Cameras":
        query["camera_no"] = camera
    
    # Alert type filter using contains logic
    if alert_type and alert_type != "All Alert Types":
        query["alert_type"] = {"$regex": alert_type, "$options": "i"}

    cursor = alerts_col.find(query, {"alert_type": 1, "date_time": 1, "area": 1, "_id": 0})
    data = list(cursor)
    
    hour_counts = {}
    types = set()

    for doc in data:
        area = doc.get("area", "")
        dt_str = doc.get("date_time", "")
        hour = None
        try:
            if " " in dt_str:
                time_part = dt_str.split(" ")[1]
                if ":" in time_part:
                    hour = int(time_part.split(":")[0])
                elif "-" in time_part:
                    hour = int(time_part.split("-")[0])
            # If no valid format, hour remains None
        except Exception:
            continue
        if hour is not None and 0 <= hour <= 23:
            # Determine area based on area
            if area.startswith("MLB"):
                area_label = "Module Manufacturing"
            elif area.startswith("CLB"):
                area_label = "Cell Manufacturing"
            else:
                area_label = area
            hour_counts.setdefault(hour, {})
            hour_counts[hour].setdefault(area_label, 0)
            hour_counts[hour][area_label] += 1
            types.add(area_label)

    hours = list(range(0, 24))
    series = {t: [hour_counts.get(h, {}).get(t, 0) for h in hours] for t in types}

    return jsonify({"ok": True, "hours": hours, "types": sorted(list(types)), "series": series})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5053, debug=True)