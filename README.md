
# Waaree Final System

This repository contains the main deployment and dashboard system for Waaree device monitoring and alerting. Below are the key components and their roles:

## Main Files & Structure

### 1. `deploy_waaree.py` (Main File)
This is the core deployment script responsible for:
- Loading camera configurations
- Scheduling and running device tasks
- Processing camera feeds for various use cases (crowd, loitering, intrusion, safety, downtime, camera angle check, etc.)
- Generating and saving alerts
- Integrating with external detection servers and APIs

### 2. `mongo.py` (MongoDB Injection)
Handles all MongoDB interactions for:
- Storing and retrieving alert and footfall data
- Database connection setup
- Utility functions for data management

### 3. `receive.py` (Receiving Data)
Responsible for:
- Receiving data from devices or external sources
- Processing incoming messages or files
- Integrating received data into the system or database

### 4. `sending_local_data.py` (Sending Data)
Handles:
- Sending processed data or alerts from the device to external endpoints or servers
- Data packaging and transmission logic

### 5. `Dashboard/` (Web Dashboard)
A Flask-based web dashboard for monitoring and visualizing device data and alerts. Includes:
- `app.py`: Main Flask application with API routes and web pages
- `templates/`: HTML templates for dashboard views (alerts, compliance, footfall, production, etc.)
- `static/`: Static assets (CSS, JS, images) for dashboard UI

#### Dashboard Features
- View and filter alerts by area, line, camera, type, and time
- Footfall and vehicle exit analytics
- Production and compliance monitoring
- Image serving and visualization

---

## Getting Started
1. Ensure MongoDB is running and accessible.
2. Configure camera and device settings in the config files.
3. Run `deploy_waaree.py` to start device monitoring and alert generation.
4. Start the dashboard with `python Dashboard/app.py` to access the web interface.

## Folder Structure
- `deploy_waaree.py` — Main deployment logic
- `mongo.py` — Database operations
- `receive.py` — Data receiving
- `sending_local_data.py` — Data sending
- `Dashboard/` — Web dashboard (Flask app)

