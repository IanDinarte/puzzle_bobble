import threading
import pyautogui
import keyboard
import time
import os
import torch
import cv2
import numpy as np
import math
from PIL import Image
from ultralytics import YOLO

def extract_pointer_angle(cv_image, pointer_box):
    x1, y1, x2, y2 = pointer_box
    pointer_crop = cv_image[y1:y2, x1:x2]

    gray = cv2.cvtColor(pointer_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No pointer contour found.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    points = approx[:, 0, :]
    tip = tuple(points[np.argmin(points[:, 1])])
    base = tuple(points[np.argmax(points[:, 1])])
    tip = (tip[0] + x1, tip[1] + y1)
    base = (base[0] + x1, base[1] + y1)

    def get_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))

    return get_angle(base, tip)

def calculate_target_angle(pointer_base, target_x):
    dx = target_x - pointer_base[0]
    dy = -100  # assume some constant upward direction
    return math.degrees(math.atan2(dy, dx))

def take_screenshot(stop_event,model):
    screenx_center = 1920 / 2
    screeny_center = 1080 / 2
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    while not stop_event.is_set():
        #detect what bubble will be shot
        target_box = torch.tensor([924.0000, 860.0000, 994.000, 942.000])
        def point_in_box(cx, cy, box):
            x1, y1, x2, y2 = box
            return x1 <= cx <= x2 and y1 <= cy <= y2
        
        screenshot = pyautogui.screenshot()
        screenshot = Image.frombytes('RGB', screenshot.size, screenshot.tobytes())
        print("screenshot took")

        results = model([screenshot], conf=.50)

        shooter_class = None
        pointer_y_threshold = 900
        min_dist = float('inf')
        target_coords = None
        
        boxes = results[0].boxes
        
        pointer_box = None
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i]  
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]

            if shooter_class is None and point_in_box(cx.item(), cy.item(), target_box):
                shooter_class = class_name
                print(f"Shooter bubble: {shooter_class}")
                continue  
            
            if pointer_box is None and class_name == "pointer":
                pointer_box = box.cpu().numpy().astype(int)
                continue
            
            if cy.item() > pointer_y_threshold:
                continue
            if shooter_class is not None and class_name != shooter_class:
                continue
            
            dist = abs(cx.item() - screenx_center)
            if dist < min_dist:
                min_dist = dist
                target_coords = cx.item()
                
        cv_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        if pointer_box is not None:
            new_pointer_angle = extract_pointer_angle(cv_image, pointer_box)
            if new_pointer_angle is not None:
                pointer_angle = new_pointer_angle
                print(f"Pointer angle: {pointer_angle:.2f}°")
        else:
            print("pointer not found, using previous angle")

        if target_coords is not None and pointer_box is not None:
            pointer_data = extract_pointer_angle(cv_image, pointer_box)
            if pointer_data is not None:
                pointer_angle = pointer_data
                print(f"Pointer angle: {pointer_angle:.2f}°")

                # Estimate base x position (center of pointer_box bottom)
                x1, y1, x2, y2 = pointer_box
                pointer_base_x = (x1 + x2) // 2
                target_angle = calculate_target_angle((pointer_base_x, y2), target_coords)

                print(f"Target angle: {target_angle:.2f}°")

                if abs(pointer_angle - target_angle) < 2:
                    print("Aligned! Shooting...")
                    pyautogui.press(" ")
                elif pointer_angle < target_angle:
                    print("Adjusting right...")
                    pyautogui.keyDown("right")
                    time.sleep(0.005)
                    pyautogui.keyUp("right")
                else:
                    print("Adjusting left...")
                    pyautogui.keyDown("left")
                    time.sleep(0.005)
                    pyautogui.keyUp("left")
        else:
            print("No valid target found, shooting randomly.")
            #pyautogui.press(" ")

        

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best.pt")
    model = YOLO(model_path)
    stop_event = threading.Event()

    screenshot_thread = threading.Thread(
        target=take_screenshot, args=(stop_event, model,)
    )
    screenshot_thread.start()
    
    keyboard.wait('q')
    
    stop_event.set()
    
    screenshot_thread.join()
    
    print("program stopped")

main()
