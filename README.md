# 🚁 Autonomous Drone AI — Object Detection & Payload Delivery

An autonomous UAV system integrating **GroundingDINO computer vision, GPS-based localization, MAVLink flight control, spatial clustering, and automated payload delivery**.

The system is designed to detect predefined objects from aerial imagery, estimate their geographic location, cluster multiple detections into target points, and autonomously navigate the drone for payload deployment.

---

## 📌 Project Overview

This project combines **AI-based aerial object detection with autonomous drone navigation** to create an end-to-end search, localization, and payload-delivery pipeline.

The system follows the workflow:

**Aerial Search → Object Detection → GPS Localization → Target Validation → Clustering → Navigation → Payload Delivery → Mission Resumption**

GroundingDINO is used for open-vocabulary detection, while MAVLink provides communication and control between the software system and the UAV.

---

## 🧠 Computer Vision

The detection pipeline uses **GroundingDINO** to identify objects from aerial images using natural-language prompts.

Example target objects include:

- 🏂 White snowboard
- 🎿 White skis
- 🏀 Small orange basketball
- 🛑 Red stop sign

### Detection Pipeline

1. Load aerial image
2. Run GroundingDINO inference
3. Extract bounding boxes, confidence scores and labels
4. Apply confidence and text thresholds
5. Apply Non-Maximum Suppression (NMS)
6. Validate detected target
7. Convert image coordinates to GPS coordinates

---

## 📍 Image-to-GPS Localization

Detected objects are converted from image-space coordinates into geographic coordinates.

The localization system uses:

- Camera waypoint coordinates
- Detection bounding-box center
- Image dimensions
- Ground Sampling Distance (GSD)
- Camera/drone geometry
- Latitude and longitude conversion

This allows an object detected in an aerial image to be associated with an estimated **real-world GPS location**.

---

## 🗺️ Geospatial Target Validation

Before a detection is considered a valid target, its estimated GPS position is checked against a predefined operational boundary.

The system uses **Shapely Polygon geometry** to ensure that detected targets fall within the permitted search area.


Aerial Detection
       ↓
Image Coordinates
       ↓
GPS Position
       ↓
Operational Boundary Check
       ↓
Valid Target



AUTO
GUIDED
RTL
