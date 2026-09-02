""" Ani's final version
mav done 
shot done 
"""


import os
import re
import csv
import cv2
import sys
import time
import math
import torch
import pickle
import shutil
import signal
import argparse
import threading
import subprocess
import multiprocessing
import datetime as dt
import numpy    as np
import pandas   as pd
from torchvision.ops  import nms
from pathlib          import Path
from pymavlink        import mavutil,mavwp
from sklearn.cluster  import DBSCAN 
from shapely.geometry import Point,Polygon
from PIL              import Image,ImageDraw,ImageFont
from groundingdino.util.inference import load_model, load_image, predict

print('\n')
# Configuration Parameters
GSD = 0.3
drop_alt = 30
drop_time = 30
threshold = 25
threshold_step = 5
bounding_box_coords = [
    (13.030947, 77.565240),
    (13.031217, 77.565284),
    (13.031195, 77.565582),
    (13.031080, 77.565680),
    (13.030896, 77.565680),
    (13.030883, 77.565623),
    (13.030843, 77.565613),
    (13.030870, 77.565319),
    (13.030905, 77.565300)
]

# Load GroundingDINO Model
model_path = "weights/groundingdino_swint_ogc.pth"
config_path = "/home/edhitha/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n[INFO] Using device:", device)

start_time = time.time()
model = load_model(config_path, model_path).to(device).eval()
model_load_time = time.time() - start_time
print("[INFO] Model loaded in {:.2f} seconds.".format(model_load_time))

print('GSD: ',GSD,'drop_alt: ',drop_alt,'drop_time: ',drop_time)
print('threshold: ',threshold,'threshold step: ',threshold_step)
print('grid: ',bounding_box_coords)

# Directories and Paths
main_dir = '/home/edhitha/AI'
os.makedirs(main_dir, exist_ok=True) 
in_imgs = '/home/edhitha/DCIM/test_cam/images'
crops_dir = main_dir + '/crops_dir'
os.makedirs(crops_dir, exist_ok=True)
pickle_path = '/home/edhitha/DCIM/test_cam/cam_wps.pickle'
csv_path = main_dir + '/results.csv'



print('checking connection')
the_connection=mavutil.mavlink_connection('udp:127.0.0.1:14666')
####the_connection=mavutil.mavlink_connection('udp:0.0.0.0:14550')############
the_connection.wait_heartbeat()
print("Heartbeat received from system (system %u component %u)" %(the_connection.target_system, the_connection.target_component))

'''------------------------------------------------READ WAYPOINTS---------------------------------------------------------'''
def get_wps_and_indicies(master):
    # Request mission items
    master.mav.mission_request_list_send(master.target_system, master.target_component)

    # Wait for the mission count message
    count_msg = master.recv_match(type='MISSION_COUNT', blocking=True)
    num_items = count_msg.count

    # Request each mission item
    commands = []
    for seq in range(num_items):
        master.mav.mission_request_send(master.target_system, master.target_component, seq)

        # Wait for the mission item message
        item_msg = master.recv_match(type='MISSION_ITEM', blocking=True)
        commands.append(item_msg)

    # Create a DataFrame to store mission commands
    columns = ['Index', 'Latitude', 'Longitude', 'Altitude', 'Command', 'Param1', 'Param2', 'Param3', 'Param4', 'AutoContinue']
    commands_df = pd.DataFrame(columns=columns)

    # Populate DataFrame with mission command data
    for cmd in commands:
        commands_df.loc[len(commands_df)] = [
            cmd.seq,
            cmd.x,  # Coordinates are in degrees * 1e7
            cmd.y,
            cmd.z,
            cmd.command,
            cmd.param1,
            cmd.param2,
            cmd.param3,
            cmd.param4,
            cmd.autocontinue
        ]


    # Print the DataFrame
    #print("Mission Commands DataFrame:")
    #print(commands_df)
    # Find the WP index of last lap waypoint
    lap_last_waypoint = commands_df[commands_df['Command'] == 177].index[0] - 1
    sg_first_waypoint = commands_df[commands_df['Command'] == 177].index[0] + 2

    # Find the lWP index of last Seacrh Grid waypoint
    search_grid_last_wp = commands_df[commands_df['Command'] == 177].index[-1] 
    second_do_jump=commands_df[commands_df['Command'] == 177].index[-1]
    print('second:',second_do_jump)
    while commands_df.loc[search_grid_last_wp]['Command'] != 16:
        print(search_grid_last_wp)
        search_grid_last_wp -= 1
    # Print the result
    result = commands_df.iloc[[lap_last_waypoint, search_grid_last_wp], [1, 2]]
    print(result)
    last_lap_wp = commands_df.iloc[lap_last_waypoint, [1, 2]].tolist()   # Latitude and Longitude for last lap wp
    lap_alt = commands_df.iloc[lap_last_waypoint, 3]                     #Altitude of last lap wp
    last_sg_wp = commands_df.iloc[search_grid_last_wp, [1, 2]].tolist()  # Latitude and Longitude for last search grid wp
    print(f"lap_last_waypoint: {lap_last_waypoint}, search_grid_last_wp: {search_grid_last_wp}, lap alt: {lap_alt}")
    return last_lap_wp, last_sg_wp, lap_alt,lap_last_waypoint,search_grid_last_wp,sg_first_waypoint

print('getting waypoints...')
last_lap,last_sg, lap_alt,lp_wp,last_sg_wp,first_sg_wp=get_wps_and_indicies(the_connection)

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.60
BOX_THRESHOLD = 0.40
TEXT_THRESHOLD = 0.35
IOU_THRESHOLD = 0.2
MAX_AREA_RATIO = 0.50
MIN_BOX_AREA_RATIO = 0.005

# Detect Prompt
text = 'white snowboard. white skis. small orange basketball. red stop sign'

'''------------------------------------------------CHANGE MODES---------------------------------------------------------'''

def guided():
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,176, 0, 1, 4, 0, 0, 0, 0, 0)
    msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
    print(msg,file=sys.stderr)

def auto():
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,176, 0, 1, 3, 0, 0, 0, 0, 0)
    msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
    print(msg,file=sys.stderr)

def loiter():
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                     176, 0, 1, 5, 0, 0, 0, 0, 0)
    msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
    print(msg,file=sys.stderr)

def guide_command(value, alt):
    the_connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(10, the_connection.target_system,the_connection.target_component, 6, 1024, int(value[0]*1e7), int(value[1]*1e7),alt , 0, 0, 0, 0, 0, 0, 0, target_bearing_int(value[0], value[1])))

'''---------------------------------------------REPOSITIONING PART------------------------------------------------------'''

# def calculate_pixel_center(x_center, y_center, img_width, img_height):
#     """Calculate pixel center from normalized coordinates."""
#     x_pixel_center = int(x_center * img_width)
#     y_pixel_center = int(y_center * img_height)
#     return x_pixel_center, y_pixel_center

def read_dict_from_file(pickle_path):
    while True:
        try:
            # Synchronous code to read from the file
            with open(pickle_path, 'rb') as file:
                loaded_dict = pickle.load(file)
            return loaded_dict
        except Exception as e:
            print('Error occurred while loading the cam_wps:', e)
            time.sleep(0.2133253665)


# def get_midpoint(image_path):
#     # Open the image
#     img = Image.open(image_path)
#     # Get the width and height of the image
#     width, height = img.size
#     # Calculate the mid_x and mid_y coordinates
#     mid_x = width // 2
#     print("in midpoint fxn",file=sys.stderr)
#     mid_y = height // 2
#     return (mid_x, mid_y)


def newlatlon(lat , lon , hdg ,dist, movementHead):
    lati=math.radians(lat)
    longi=math.radians(lon)
    rade = 6367489
    AD = dist/rade
    print("5",file=sys.stderr)
    sumofangles = (hdg + movementHead)%360
    newheading = math.radians(sumofangles)
    newlati = math.asin(math.sin(lati)*math.cos(AD) + math.cos(lati)*math.sin(AD)*math.cos(newheading))
    newlongi = longi + math.atan2(math.sin(newheading)*math.sin(AD)*math.cos(lati), math.cos(AD)-math.sin(lati)*math.sin(newlati))
    ret_lat = int(round(math.degrees(newlati*1e7)))
    ret_lon = int(round(math.degrees(newlongi*1e7)))
    return ret_lat, ret_lon


def get_coordinates(image_name,image_path,pickle_path,point2,point1, lat=None, lon=None, yaw=None):
    if lat is None or lon is None or yaw is None:
        wps_list = read_dict_from_file(pickle_path)
        index = int(image_name[-9:-4])
        print(wps_list[index])
        if len(wps_list[index]) == 6:
            lat, lon, alt, head, yaw, time1 = wps_list[index]
        else:
            lat, lon, alt, time1 = wps_list[index]
            yaw = 165.31
    
    if lat is not None and lon is not None:
        pixel_distance = math.sqrt((point2[0] - point1[0]) * 2 + (point2[1] - point1[1]) * 2)
        angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0])) - 90
        real_distance_gsd = (pixel_distance * GSD) / 100

        if lat == 0:
            print(f"NO LAT AND LON FOUND, {image_path}", file=sys.stderr)

        value = newlatlon(lat, lon, yaw, real_distance_gsd, angle)
        return value
    else:
        return None,None,None

# Function to calculate the angle between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.atan2(y, x)
    return math.degrees(bearing)

# Function to handle incoming messages
def target_bearing_int(target_lat, target_lon):
        msg = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        lat, lon = msg.lat / 1e7, msg.lon / 1e7
        yaw = msg.hdg / 100
        # Print heading and yaw
        if yaw is not None:
            # print(f"Heading: {heading}, Yaw: {yaw}, lat: {lat}, lon: {lon}")
            # Calculate and adjust yaw to point towards target
            if lat is not None and lon is not None:
                target_bearing = (calculate_bearing(lat, lon, target_lat, target_lon)+180) % 360
                yaw_error = target_bearing - yaw
                # if yaw_error > 180:
                #     yaw_error -= 360
                # elif yaw_error < -180:
                #     yaw_error += 360
                print("yaw:", yaw_error, target_bearing)
                return int(0)


def distance_lat_lon(lat1, lon1, lat2, lon2):
    '''distance between two points'''
    dLat = math.radians(lat2) - math.radians(lat1)
    dLon = math.radians(lon2) - math.radians(lon1)
    a = math.sin(0.5*dLat)*2 + math.sin(0.5*dLon)*2 * math.cos(lat1) * math.cos(lat2)
    c = 2.0 * math.atan2(math.sqrt(abs(a)), math.sqrt(abs(1.0-a)))
    ground_dist = 6371 * 1000 * c
    return ground_dist

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout occurred")

def execute_with_timeout(func, timeout, *args, **kwargs):
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Set the timeout alarm
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        print("Function execution timed out")
        result = None
    finally:
        # Reset the alarm
        signal.alarm(0)
    return result
    
def change_alti(current_pos,last_lap,alt,descend=False,ascend=False):
    if descend:  #alt=drop_alt, lat_lon=last wp of lap (last_lap_wp)
        print(f'Dropping alt to: {alt}m',file=sys.stderr)
        while True:
             try:
                 execute_with_timeout(guided, 2)
                 time.sleep(1)
                 guide_command(last_lap, alt)
                 break
             except Exception as e:
                 print("Timeout Exception")
                 pass
        try:
            while True:
                msg=the_connection.recv_match(type=['GLOBAL_POSITION_INT'],blocking=True)
                current_alt=msg.alt
                alti=current_alt/1e3
                alt_dist=alti-alt
                print('altitude: ',alt_dist)
                if alt_dist<=1:
                    return True
        except Exception as e:
                print(e)
    pass
    if ascend:  #alt=lap_alt, lat_lon=latest executed drop lat,lon (value)
        print(f'Raising alt to: {alt}m',file=sys.stderr)
        while True:
            try:
                execute_with_timeout(guided, 2)
                time.sleep(1)
                guide_command(current_pos, alt)
                break
            except Exception as e:
                print("Timeout Exception")
                pass
        try:
            while True:
                msg=the_connection.recv_match(type=['GLOBAL_POSITION_INT'],blocking=True)
                current_alt=msg.alt
                alti=current_alt/1e3
                alt_dist=alt-alti
                print('altitude: ',alt_dist)
                if alt_dist<=1:
                    return True
        except Exception as e:
                print(e)

def automation(value,target):
    print("target aquired",file=sys.stderr)
    # print('Value:', value)
    gps = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking = True)
    # print(gps)
    calculated_distance =  distance_lat_lon(value[0],value[1],gps.lat/1e7,gps.lon/1e7)

    kurrentalti = 30
    accuracy = 1

    # print(kurrentalti)
    print('changing Guided',file=sys.stderr)
    while True:
        try:
            execute_with_timeout(guided, 2)
            time.sleep(1)
            guide_command(value, drop_alt)
            break
        except Exception as e:
            print("Timeout Exception")
            pass
    
    # msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
    print('Desired Latitude and Longitude sent ....', file=sys.stderr)
    print('Desired lat and lon:', value[0], value[1])
    print("Real_distance between lats and lons", calculated_distance,file=sys.stderr)
    # Wait until the drone reaches the target location
    try:
        while True:
            msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT'], blocking=True)
            current_lat = msg.lat / 1e7
            current_lon = msg.lon / 1e7
            distance = distance_lat_lon(current_lat,current_lon,value[0],value[1])
            print('Distance between drone and target:',distance,'for lat and lon:', current_lat, current_lon, end="\r")
            if distance <= accuracy:
                print('Started Dropping ...',file=sys.stderr)
                drop(target)
                
                the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,181, 0, 0, 1, 0, 0, 0, 0, 0)
                time.sleep(drop_time)
                    
                print('dropped',file=sys.stderr)
                print('Current drop loc latitude:', msg.lat / 1e7,file=sys.stderr)
                print('Current drop loc longtitude:', msg.lon / 1e7,file=sys.stderr)
                auto()
                break
    except Exception as e:
        print(e)
        pass


def drop(index):
    global rem
    # Set the system and component ID (replace with your system and component ID)
    system_id = the_connection.target_system
    component_id = the_connection.target_component

    # Set the auxiliary pin or relay channel (replace 0 with your channel number) (Channel, PWM_VALUE)
    pwm_values = [(8,2000),(8,1000),(7,1000),(7,2000)]
    # Set the initial PWM value
    pwm_value = pwm_values[index-1]
    print(pwm_value)
    # Set the MAV_CMD_DO_SET_SERVO command parameters
    command = mavutil.mavlink.MAV_CMD_DO_SET_SERVO

    param1 = pwm_value[0]
    param2 = pwm_value[1]

    # Send the MAV_CMD_DO_SET_SERVO command
    the_connection.mav.command_long_send(
        system_id, component_id,
        command,
        0,  # Confirmation
        param1, param2, 0, 0, 0, 0, 0
    )

def cluster(latitude, longitude, threshold, threshold_step,main_dir, use_grid):
    print('starting clustering: ')
    # threshold = maximum threshold value = 2 * drop boundary (in feet)
    # threshold_step = decrease for more accuracy

    # Define clustering parameters
    thr_main = threshold  # Maximum threshold value
    thr = threshold_step  # Increment threshold value

    # Create an initial CSV file with latitude, longitude
    initial_data = pd.DataFrame({
        "Latitude": latitude,
        "Longitude": longitude,
    })

     ######################################   use google maps and make bounding box so all targets will be detected     ###################################### 
    # Define the bounding box corners
    if use_grid:
        bounding_box = Polygon(bounding_box_coords)

        # Combine latitude and longitude into a single array
        data = np.column_stack((latitude, longitude))

        # Filter points inside the bounding box
        global filtered_data
        filtered_data = np.array([point for point in data if bounding_box.contains(Point(point[0], point[1]))])

        if filtered_data.size == 0:
            print("No points are inside the bounding box.")
            return None
    else:
        filtered_data = np.column_stack((latitude, longitude))

    all_centroids = []  # Store centroids for all thresholds
    while thr <= thr_main:
        feet_to_degrees_lat = thr / 364000
        feet_to_degrees_lon = thr / 288200
        eps = np.mean([feet_to_degrees_lat, feet_to_degrees_lon])

        dbscan = DBSCAN(eps=eps, min_samples=1)
        dbscan.fit(filtered_data)

        labels = dbscan.labels_
        unique_labels = np.unique(labels)

        centroids = [filtered_data[labels == label].mean(axis=0) for label in unique_labels]
        centroids = np.array(centroids)
        all_centroids.append(centroids)

        filtered_data = centroids  # Update data for the next iteration
        #print(f"Threshold: {thr}, Clusters: {len(centroids)}")
        thr += threshold_step

    #print("Final centroids:", centroids)

    # Save centroids
    with open(main_dir+"/clustered_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Latitude", "Longitude"])
        writer.writerows(centroids)

    print("Centroids saved to 'clustered_results.csv'")
    return centroids

cl=['Image Name','Image path','crop name','crop path','pxl center x y','lat','lon','class']
results_df=pd.DataFrame(columns=cl)

new_data = {
    'Image Name': None,
    'Image path': None,
    'crop name': None,
    'crop path': None,
    'pxl center x y': None,
    'lat': None,
    'lon': None,
    'class':None
}


def main():
    num_files = len([f for f in os.listdir(in_imgs) if os.path.isfile(os.path.join(in_imgs, f))])
    print(f"Number of images: {num_files}")
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)  
    except IOError:
        font = ImageFont.load_default()  
    
    img_files = sorted(os.listdir(in_imgs))
    lat, lon, yaw = None, None, None
    
    # Columns for results DataFrame
    cl = ['Image Name','Image path','crop name','crop path','pxl center x y','lat','lon','class']
    results_df = pd.DataFrame(columns=cl)
    
    new_data = {
        'Image Name': None,
        'Image path': None,
        'crop name': None,
        'crop path': None,
        'pxl center x y': None,
        'lat': None,
        'lon': None,
        'class': None
    }

    for image_file in img_files[:-4]:
        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            image_path = os.path.join(in_imgs, image_file)
            print("\n[INFO] Processing image:", image_file)

            start_image_time = time.time()

            # Load image for GroundingDINO
            image_source, image = load_image(image_path)
            image = image.to(device)

            original_image = Image.open(image_path).convert("RGB")
            img_width, img_height = original_image.size
            image_area = img_width * img_height
            imgXY = (img_width/2, img_height/2)
            print("   - Image Dimensions: {}x{}".format(img_width, img_height))

            # Perform object detection
            boxes, logits, phrases = predict(model, image, text, BOX_THRESHOLD, TEXT_THRESHOLD, device)

            # Convert to numpy for further processing
            all_scores = logits.cpu().numpy()
            all_boxes = boxes.cpu().numpy()
            all_labels = phrases

            if len(all_boxes) == 0:
                print("   [WARNING] No detections found in", image_file)
                continue

            # Apply Non-Maximum Suppression
            keep_indices = nms(torch.tensor(all_boxes).to(device), 
                               torch.tensor(all_scores).to(device), 
                               IOU_THRESHOLD)
            
            filtered_boxes = [all_boxes[i] for i in keep_indices]
            filtered_labels = [all_labels[i] for i in keep_indices]
            filtered_scores = [all_scores[i] for i in keep_indices]

            final_boxes, final_labels, final_scores = [], [], []
            for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
                x_center, y_center, width, height = box

                # Convert to x1, y1, x2, y2
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height

                # Ensure valid bounding box dimensions
                x1, x2 = max(0, min(img_width, x1)), max(0, min(img_width, x2))
                y1, y2 = max(0, min(img_height, y1)), max(0, min(img_height, y2))

                # Calculate bounding box area
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / image_area

                if (score >= CONFIDENCE_THRESHOLD and 
                    MIN_BOX_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO and 
                    (x2 - x1) > 20 and (y2 - y1) > 20):
                    
                    final_boxes.append([x1, y1, x2, y2])
                    final_labels.append(label)
                    final_scores.append(score)

            if not final_boxes:
                print("   [INFO] No final detections remaining after filtering for", image_file)
                continue

            draw = ImageDraw.Draw(original_image)
            for idx, (box, label, score) in enumerate(zip(final_boxes, final_labels, final_scores)):
                x1, y1, x2, y2 = box
                label_text = "{} ({:.2f})".format(label, score)
                
                draw.rectangle(box, outline="blue", width=4)  
                draw.text((x1, y1 - 10), label_text, fill="blue", font=font)  
                
                crop = original_image.crop((x1, y1, x2, y2))
                crop_filename = "{}crop{}.jpg".format(image_file.split('.')[0], idx)
                crop_path = os.path.join(crops_dir, crop_filename)
                crop.save(crop_path)
                print("   - Crop saved:", crop_path)
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cropXY = (center_x, center_y)

                try:
                    value = get_coordinates(image_file, image_path, pickle_path, imgXY, cropXY, lat, lon, yaw)
                except IndexError:
                    print('image does not contain metadata: ', image_file)
                    break

                lat = value[0]/1e7
                lon = value[1]/1e7
                
                new_data['Image Name'] = image_file
                new_data['Image path'] = image_path
                new_data['crop name'] = crop_filename
                new_data['crop path'] = crop_path
                new_data['pxl center x y'] = cropXY
                new_data['lat'] = lat
                new_data['lon'] = lon           
                new_data['class'] = label_text
                
                results_df.loc[len(results_df.index)] = new_data

            process_time = time.time() - start_image_time
            print("   - Processing time: {:.2f} seconds".format(process_time))
    
    results_df.to_csv(csv_path, index=False)
    return results_df

try:
    while True:
        m = the_connection.recv_match(type=['MISSION_ITEM_REACHED'], blocking=True)
        print('waypoint: ', m.seq)
        if m.seq == last_sg_wp:
            guided()
            print('stop geotagging...')
            time.sleep(5)
            df = main()
            break
    
    print('starting reposition')
    latitude = df['lat']
    longitude = df['lon']
    final_centroids = cluster(latitude, longitude, threshold, threshold_step, main_dir, use_grid=True)
    
    if final_centroids is not None:
        lati, longi = final_centroids[:, 0], final_centroids[:, 1]
    
    print(lati)
    print(longi)

    for i in range(0, 4):
        if i == 0:
            print(f'\n target: {i+1}')
            val = [lati[i], longi[i]]
            print(val)
            automation(val, i+1)
        else:
            try:
                while True:
                    msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT'], blocking=True)
                    m = the_connection.recv_match(type=['MISSION_ITEM_REACHED'], blocking=True)
                    current_lat = msg.lat/1e7
                    current_lon = msg.lon/1e7
                    
                    print('waypoint: ', m.seq)
                    
                    if m.seq == lp_wp:
                        print(f'\n target: {i+1}')
                        val = [lati[i], longi[i]]
                        print(val)
                        time.sleep(1)
                        automation(val, i+1)
                        break
            except Exception as e:
                print(e)
                pass
except Exception as e:
    print(e)
    pass
