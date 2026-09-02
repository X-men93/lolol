# GroundingDINO-based zero-shot object detection using Hugging Face Transformers.
# Detects specified objects from input images using text prompts.
# Applies confidence, IoU/NMS, and bounding-box area filtering.
# Draws bounding boxes and confidence scores on detected objects.
# Saves object crops and detection coordinates.
# Exports detection results with image name, label, confidence, and center coordinates to CSV.
# Automatically uses CUDA GPU if available, otherwise CPU.
import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import nms  

# Load model
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n[INFO] Using device:", device)

start_time = time.time()
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
model_load_time = time.time() - start_time
print("[INFO] Model and processor loaded in {:.2f} seconds.".format(model_load_time))

# Input and output directories
image_folder = "/home/pushpa/yolo/GroundingDINO/input"
output_folder = "/home/pushpa/yolo/GroundingDINO/output"
crop_folder = "/home/pushpa/yolo/GroundingDINO/detections_crops"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(crop_folder, exist_ok=True)

# Detection description
text = "a white aeroplane. two white strips together. a white snowboard. a brownish suitcase. a grey bed/mattress with white tag on it."

# Threshold settings
CONFIDENCE_THRESHOLD = 0.3  
BOX_THRESHOLD = 0.35  
TEXT_THRESHOLD = 0.40  
IOU_THRESHOLD = 0.2  
AREA_THRESHOLD = 0.6  
MIN_BOX_AREA_RATIO = 0.005  

# Load font for drawing text
try:
    font = ImageFont.truetype("arial.ttf", 30)  
except IOError:
    font = ImageFont.load_default()  

# CSV storage
csv_data = []

print("\n[INFO] Processing images in:", image_folder)
image_list = sorted(os.listdir(image_folder))

if not image_list:
    print("[WARNING] No images found in the input directory. Exiting program.")
    exit()

for image_file in image_list:
    if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        image_path = os.path.join(image_folder, image_file)
        print("\n[INFO] Processing image:", image_file)

        start_image_time = time.time()

        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        image_area = img_width * img_height  
        print("   - Image Dimensions: {}x{}".format(img_width, img_height))

        inputs = processor(images=image, text=text, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]]
        )

        if not results:
            print("   [WARNING] No detections found in", image_file)
            continue

        all_boxes, all_labels, all_scores = [], [], []
        for result in results:
            all_boxes.extend(result["boxes"])
            all_labels.extend(result["labels"])
            all_scores.extend(result["scores"])

        if not all_boxes:
            print("   [WARNING] No valid detections found after processing in", image_file)
            continue

        all_boxes_tensor = torch.stack(all_boxes).to(device)
        all_scores_tensor = torch.tensor(all_scores).to(device)

        keep_indices = nms(all_boxes_tensor, all_scores_tensor, IOU_THRESHOLD)

        filtered_boxes = [all_boxes[i] for i in keep_indices]
        filtered_labels = [all_labels[i] for i in keep_indices]
        filtered_scores = [all_scores[i] for i in keep_indices]

        print("   - Total Detections before filtering:", len(filtered_boxes))

        final_boxes, final_labels, final_scores = [], [], []
        for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
            box = box.tolist()  
            box_area = (box[2] - box[0]) * (box[3] - box[1])  

            if score >= CONFIDENCE_THRESHOLD and MIN_BOX_AREA_RATIO <= box_area / image_area <= AREA_THRESHOLD:
                final_boxes.append(box)
                final_labels.append(label)
                final_scores.append(score)
            else:
                print("   [REMOVED] {} | Score: {:.2f} | Area: {:.2%}".format(label, score, box_area / image_area))

        print("   - Detections after filtering:", len(final_boxes))

        if not final_boxes:
            print("   [INFO] No final detections remaining after filtering for", image_file)
            continue

        draw = ImageDraw.Draw(image)
        for idx, (box, label, score) in enumerate(zip(final_boxes, final_labels, final_scores)):
            label_text = "{} ({:.2f})".format(label, score)
            draw.rectangle(box, outline="blue", width=4)  
            draw.text((box[0], box[1] - 10), label_text, fill="blue", font=font)  

            crop = image.crop((box[0], box[1], box[2], box[3]))
            crop_filename = "{}_crop_{}.jpg".format(image_file.split('.')[0], idx)
            crop_path = os.path.join(crop_folder, crop_filename)
            crop.save(crop_path)
            print("   - Crop saved:", crop_path)

            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            csv_data.append([image_file, label, score, center_x, center_y])

        output_path = os.path.join(output_folder, image_file)
        image.save(output_path)
        print("   - Processed image saved:", output_path)

        process_time = time.time() - start_image_time
        print("   - Processing time: {:.2f} seconds".format(process_time))

csv_path = "/home/pushpa/yolo/GroundingDINO/detections.csv"
df = pd.DataFrame(csv_data, columns=["Image Name", "Detection Name", "Confidence Score", "Center X", "Center Y"])
df.to_csv(csv_path, index=False)
print("\n[INFO] CSV file saved:", csv_path)

print("\n[INFO] Processing complete. Check output folders for results.")
