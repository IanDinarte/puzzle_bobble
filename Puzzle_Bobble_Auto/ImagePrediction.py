import os
import torch
from PIL import Image
from ultralytics import YOLO

target_box = torch.tensor([924.0000, 860.0000, 994.000, 942.000])

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")

# Load a model
model = YOLO(model_path)  # pretrained YOLOv8n model

image_path = os.path.join(script_dir, "test4.png")

image = Image.open(image_path)

#image = Image.frombytes("RGB", image.size, image.tobytes())

results = model([image], conf=0.80, save=True)  # return a list of Results objects

class_names = model.names

def point_in_box(cx, cy, box):
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2

boxes = results[0].boxes
for i in range(len(boxes)):
    box = boxes.xyxy[i]  # (x1, y1, x2, y2)
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    class_id = int(boxes.cls[i].item())
    
    if point_in_box(cx.item(), cy.item(), target_box):
        print(f"{class_names[class_id]} is in the target location at center ({cx.item():.1f}, {cy.item():.1f})")