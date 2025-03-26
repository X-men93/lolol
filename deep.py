#!/usr/bin/env python3
"""
Autonomous Drone Payload Delivery System with Enhanced Error Handling
"""

import os
import sys
import time
import math
import logging
import pickle
import signal
import numpy as np
from pathlib import Path
from pymavlink import mavutil
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon
from groundingdino.util.inference import load_model, load_image, predict
import torch
from torchvision.ops import nms

# ---------------------------- Configuration ---------------------------- #
CONFIG = {
    "DETECTION_CAPTION": "white snowboard. white skis. small orange basketball. red stop sign",
    "DETECTION": {
        "CONFIDENCE": 0.60,
        "BOX_THRESH": 0.40,
        "TEXT_THRESH": 0.35,
        "NMS_IOU": 0.2,
        "MAX_AREA": 0.50,
        "MIN_AREA": 0.005
    },
    "GSD": 0.3,
    "DROP_ALT": 30,
    "DROP_TIME": 30,
    "CLUSTER_THRESHOLD": 25,
    "THRESHOLD_STEP": 5,
    "POSITION_ACCURACY": 3.0,
    "NAV_TIMEOUT": 30,
    "BOUNDING_BOX": [
        (13.030947, 77.565240),
        (13.031217, 77.565284),
        (13.031195, 77.565582),
        (13.031080, 77.565680),
        (13.030896, 77.565680),
        (13.030883, 77.565623),
        (13.030843, 77.565613),
        (13.030870, 77.565319),
        (13.030905, 77.565300)
    ],
    "PATHS": {
        "MAIN_DIR": "/home/edhitha/AI",
        "IMAGE_DIR": "/home/edhitha/DCIM/test_cam/images",
        "CROP_DIR": "crops",
        "PICKLE_FILE": "/home/edhitha/DCIM/test_cam/cam_wps.pickle",
        "RESULTS_CSV": "detection_results.csv",
        "MODEL": {
            "CONFIG": "/home/edhitha/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "WEIGHTS": "weights/groundingdino_swint_ogc.pth"
        }
    }
}

# ---------------------------- Logging Setup ---------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("operation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------- MAVLink Controller ---------------------------- #
class DroneController:
    def __init__(self, connection_str):
        self.connection = None
        self.current_mode = "UNKNOWN"
        self.position = {'lat': 0, 'lon': 0, 'alt': 0, 'hdg': 0}
        self._connect(connection_str)

    def _connect(self, conn_str):
        """Establish MAVLink connection with retries"""
        max_retries = 5
        for attempt in range(1, max_retries+1):
            try:
                logger.info(f"Connection attempt {attempt}/{max_retries}")
                self.connection = mavutil.mavlink_connection(conn_str)
                self.connection.wait_heartbeat()
                logger.info("Connection established")
                return
            except Exception as e:
                logger.error(f"Connection failed: {str(e)}")
                if attempt == max_retries:
                    raise ConnectionError("Max connection attempts reached")
                time.sleep(2 ** attempt)

    def update_position(self):
        """Update current position from MAVLink messages"""
        try:
            msg = self.connection.recv_match(
                type='GLOBAL_POSITION_INT',
                blocking=True,
                timeout=5
            )
            if msg:
                self.position = {
                    'lat': msg.lat / 1e7,
                    'lon': msg.lon / 1e7,
                    'alt': msg.alt / 1e3,
                    'hdg': msg.hdg / 100
                }
        except Exception as e:
            logger.error(f"Position update failed: {str(e)}")

    def change_mode(self, mode):
        """Change flight mode with confirmation"""
        mode = mode.upper()
        mode_id = mavutil.mode_mapping_acm.get(mode)
        if not mode_id:
            logger.error(f"Invalid mode: {mode}")
            return False

        try:
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id, 0, 0, 0, 0, 0
            )
            ack = self.connection.recv_match(
                type='COMMAND_ACK',
                blocking=True,
                timeout=3
            )
            if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                self.current_mode = mode
                return True
            return False
        except Exception as e:
            logger.error(f"Mode change failed: {str(e)}")
            return False

    def navigate_to(self, lat, lon, alt):
        """Send navigation command to specified coordinates"""
        try:
            self.connection.mav.send(
                mavutil.mavlink.MAVLink_set_position_target_global_int_message(
                    10,
                    self.connection.target_system,
                    self.connection.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    0b1111111111111000,
                    int(lat * 1e7),
                    int(lon * 1e7),
                    alt * 1000,
                    0, 0, 0, 0, 0, 0, 0, 0
                )
            )
            return True
        except Exception as e:
            logger.error(f"Navigation command failed: {str(e)}")
            return False

# ---------------------------- Image Processor ---------------------------- #
class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.waypoints = self._load_waypoints()
        self.detections = []
        self.processed_images = set()

    def _load_model(self):
        """Load and initialize detection model"""
        try:
            model_path = Path(self.config["PATHS"]["MODEL"]["WEIGHTS"])
            config_path = Path(self.config["PATHS"]["MODEL"]["CONFIG"])
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model weights missing: {model_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"Model config missing: {config_path}")

            logger.info("Loading detection model...")
            start_time = time.time()
            model = load_model(str(config_path), str(model_path))
            model = model.to(self.device).eval()
            logger.info(f"Model loaded in {time.time()-start_time:.2f}s")
            return model
        except Exception as e:
            logger.critical(f"Model loading failed: {str(e)}")
            sys.exit(1)

    def _load_waypoints(self):
        """Load camera waypoints from pickle file"""
        try:
            with open(self.config["PATHS"]["PICKLE_FILE"], "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.critical(f"Waypoint loading failed: {str(e)}")
            sys.exit(1)

    def process_image(self, image_path):
        """Process single image and store detections"""
        try:
            img_path = Path(image_path)
            if img_path.name in self.processed_images:
                return

            logger.info(f"Processing {img_path.name}")
            image_source, image = load_image(str(img_path))
            image = image.to(self.device)

            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=self.config["DETECTION_CAPTION"],
                box_threshold=self.config["DETECTION"]["BOX_THRESH"],
                text_threshold=self.config["DETECTION"]["TEXT_THRESH"],
                device=self.device
            )

            valid_detections = self._filter_detections(
                boxes.cpu().numpy(),
                logits.cpu().numpy(),
                phrases,
                image_source,
                str(img_path)
            )
            
            self.detections.extend(valid_detections)
            self.processed_images.add(img_path.name)

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")

    def _filter_detections(self, boxes, scores, phrases, image_source, img_path):
        """Filter and validate detections"""
        valid = []
        try:
            keep_indices = nms(
                torch.tensor(boxes),
                torch.tensor(scores),
                self.config["DETECTION"]["NMS_IOU"]
            )

            for idx in keep_indices:
                if scores[idx] < self.config["DETECTION"]["CONFIDENCE"]:
                    continue
                
                position = self._calculate_position(
                    boxes[idx],
                    img_path,
                    image_source.shape[1],
                    image_source.shape[0]
                )
                
                if position and self._in_operational_area(position):
                    valid.append({
                        'position': position,
                        'score': scores[idx],
                        'label': phrases[idx],
                        'image': Path(img_path).name
                    })
        except Exception as e:
            logger.error(f"Detection filtering failed: {str(e)}")
        return valid

    def _calculate_position(self, box, img_path, width, height):
        """Convert detection to GPS coordinates"""
        try:
            index = int(Path(img_path).name[-9:-4])
            wp = self.waypoints[index]
            
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            dx = x_center - (width / 2)
            dy = y_center - (height / 2)
            
            distance = (math.hypot(dx, dy) * self.config["GSD"]) / 100
            angle = math.degrees(math.atan2(dy, dx)) - 90
            
            lat = wp[0] + (distance * math.cos(math.radians(angle))) / 111139
            lon = wp[1] + (distance * math.sin(math.radians(angle))) / (111139 * math.cos(math.radians(wp[0])))
            
            return (lat, lon)
        except Exception as e:
            logger.error(f"Position calculation failed: {str(e)}")
            return None

    def _in_operational_area(self, position):
        """Check if position is within bounds"""
        try:
            point = Point(position[1], position[0])
            polygon = Polygon(self.config["BOUNDING_BOX"])
            return polygon.contains(point)
        except Exception as e:
            logger.error(f"Geospatial check failed: {str(e)}")
            return False

    def get_filtered_detections(self, exclude_last=5):
        """Get detections excluding recent images"""
        try:
            sorted_images = sorted(
                self.processed_images,
                key=lambda x: os.path.getmtime(
                    os.path.join(self.config["PATHS"]["IMAGE_DIR"], x)
                ),
                reverse=True
            )
            return [d for d in self.detections if d['image'] in sorted_images[exclude_last:]]
        except Exception as e:
            logger.error(f"Detection filtering failed: {str(e)}")
            return []

# ---------------------------- Mission Manager ---------------------------- #
class MissionManager:
    def __init__(self, drone, processor):
        self.drone = drone
        self.processor = processor
        self.operation_active = True
        self.current_phase = "SEARCH_GRID"

    def execute_mission(self):
        """Main mission control loop"""
        try:
            logger.info("Starting mission execution")
            while self.operation_active:
                self._monitor_messages()
                self._process_images()
                
                if self.current_phase == "PAYLOAD_DELIVERY":
                    self._execute_payload_delivery()
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._safe_shutdown()
        except Exception as e:
            logger.critical(f"Mission failed: {str(e)}")
            self._emergency_procedures()

    def _monitor_messages(self):
        """Process incoming MAVLink messages"""
        try:
            msg = self.drone.connection.recv_match(blocking=False)
            if msg and msg.get_type() == 'MISSION_ITEM_REACHED':
                self._handle_waypoint_reached(msg.seq)
        except Exception as e:
            logger.error(f"Message monitoring failed: {str(e)}")

    def _handle_waypoint_reached(self, seq):
        """Handle waypoint reached events"""
        if seq == CONFIG["LAST_SEARCH_WP"]:
            logger.info("Final search waypoint reached")
            self.current_phase = "PAYLOAD_DELIVERY"

    def _process_images(self):
        """Process new images in directory"""
        try:
            img_dir = Path(CONFIG["PATHS"]["IMAGE_DIR"])
            for img_file in img_dir.glob("*.*"):
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.processor.process_image(str(img_file))
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")

    def _execute_payload_delivery(self):
        """Execute payload delivery sequence"""
        try:
            detections = self.processor.get_filtered_detections(5)
            if not detections:
                logger.warning("No valid detections for payload delivery")
                return

            centroids = self._cluster_detections(detections)
            for idx, (lat, lon) in enumerate(centroids):
                self._execute_drop(lat, lon, idx+1)
            
            self._return_to_mission()
        except Exception as e:
            logger.error(f"Payload delivery failed: {str(e)}")
            self._emergency_procedures()

    def _cluster_detections(self, detections):
        """Cluster detections using DBSCAN"""
        try:
            points = np.array([[d['position'][0], d['position'][1]] for d in detections])
            threshold = CONFIG["CLUSTER_THRESHOLD"]
            
            while threshold > 0:
                eps = threshold / 111139  # Convert meters to degrees
                db = DBSCAN(eps=eps, min_samples=1).fit(points)
                labels = db.labels_
                
                centroids = []
                for label in set(labels):
                    if label == -1:
                        continue
                    cluster_points = points[labels == label]
                    centroids.append(cluster_points.mean(axis=0))
                
                if centroids:
                    return centroids
                threshold -= CONFIG["THRESHOLD_STEP"]
            return []
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return []

    def _execute_drop(self, lat, lon, target_id):
        """Execute single payload drop"""
        try:
            if not self.drone.change_mode("GUIDED"):
                raise RuntimeError("Failed to enter GUIDED mode")
            
            self.drone.navigate_to(lat, lon, CONFIG["DROP_ALT"])
            start_time = time.time()
            
            while time.time() - start_time < CONFIG["NAV_TIMEOUT"]:
                self.drone.update_position()
                current = self.drone.position
                distance = self._haversine(
                    current['lat'], current['lon'], lat, lon
                )
                
                if distance <= CONFIG["POSITION_ACCURACY"]:
                    self._activate_payload(target_id)
                    time.sleep(CONFIG["DROP_TIME"])
                    return True
                time.sleep(1)
            
            raise TimeoutError("Navigation timeout")
        except Exception as e:
            logger.error(f"Drop failed: {str(e)}")
            return False

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate distance between coordinates"""
        R = 6371000  # Earth radius in meters
        φ1 = math.radians(lat1)
        φ2 = math.radians(lat2)
        Δφ = math.radians(lat2 - lat1)
        Δλ = math.radians(lon2 - lon1)

        a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _activate_payload(self, channel):
        """Activate payload mechanism"""
        try:
            self.drone.connection.mav.command_long_send(
                self.drone.connection.target_system,
                self.drone.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                channel,
                2000, 0, 0, 0, 0, 0
            )
        except Exception as e:
            logger.error(f"Payload activation failed: {str(e)}")

    def _return_to_mission(self):
        """Return to autonomous mission"""
        if self.drone.change_mode("AUTO"):
            self.current_phase = "SEARCH_GRID"
        else:
            self._emergency_procedures()

    def _emergency_procedures(self):
        """Handle emergency situations"""
        logger.critical("Initiating emergency procedures")
        self.drone.change_mode("RTL")
        self.operation_active = False

    def _safe_shutdown(self):
        """Graceful shutdown sequence"""
        logger.info("Initiating safe shutdown")
        self.drone.change_mode("RTL")
        self.operation_active = False

# ---------------------------- Main Execution ---------------------------- #
if __name__ == "__main__":
    try:
        logger.info("Initializing system components")
        drone = DroneController("udp:127.0.0.1:14666")
        processor = ImageProcessor(CONFIG)
        mission = MissionManager(drone, processor)
        
        logger.info("Starting main mission")
        mission.execute_mission()
        
    except Exception as e:
        logger.critical(f"System initialization failed: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Mission terminated")
