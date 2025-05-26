import threading
import pyautogui
import keyboard
import time
import os
import torch
import math
from PIL import Image
from ultralytics import YOLO

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best.pt")
    model = YOLO(model_path)
    stop_event = threading.Event()
    
    screenshot_thread = threading.Thread(
        target=take_screenshot,
        args=(
            stop_event,
            model,
        ),
    )
    screenshot_thread.start()

    keyboard.wait("q")

    stop_event.set()

    screenshot_thread.join()

    print("program stopped")

def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def get_class_in_location(boxes, model, target_box):
    x1, y1, x2, y2 = target_box
    centers = (boxes.xyxy[:, 0:2] + boxes.xyxy[:, 2:4]) / 2
    cx = centers[:, 0]
    cy = centers[:, 1]

    mask = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)

    if mask.any():
        index = torch.nonzero(mask)[0].item()
        target_class_id = int(boxes.cls[index].item())
        return model.names[target_class_id]
    else:
        return None

def get_shooter_class(boxes, model, shooter_box):

    shooter_class = get_class_in_location(boxes, model, shooter_box)

    if shooter_class is not None:
        print(f"Shooter bubble: {shooter_class}")
    else:
        print(f"Shooter bubble not found, using previous: {shooter_class}")

    return shooter_class

def aim_shoot(pointer_angle, target_angle):
    ANGLE_TOLERANCE = 6.0

    angle_diff = (target_angle - pointer_angle + 180) % 360 - 180
    press_duration = min(0.00001 * abs(angle_diff), 0.1)

    if abs(angle_diff) <= ANGLE_TOLERANCE:
        print("Aligned! Shooting...")
        pyautogui.press(" ")
        target_angle = None
        time.sleep(0.5)
    elif angle_diff > 0 and pointer_angle <= 160:
        print(f"Adjusting right...{press_duration}")
        pyautogui.keyDown("right")
        time.sleep(press_duration)
        pyautogui.keyUp("right")
    elif angle_diff < 0 and pointer_angle >= -160:
        print(f"Adjusting left...{press_duration}")
        pyautogui.keyDown("left")
        time.sleep(press_duration)
        pyautogui.keyUp("left")
        
    return target_angle


def take_screenshot(stop_event, model):
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    old_shooter_class = None
    shooter_class = None
    target_coords = None
    target_class = None
    target_box = None
    target_angle = None
    Y_THRESHOLD = 830
    SHOOTER_BOX = torch.tensor([924.0000, 860.0000, 994.000, 942.000])
    pointer_center = (955, 894)
    pointer_angle = 0.0

    while not stop_event.is_set():
        screenshot = pyautogui.screenshot()
        screenshot = Image.frombytes("RGB", screenshot.size, screenshot.tobytes())
        print("screenshot took")
        
        boxes = model([screenshot], conf=0.60, save=True, project='results', name=f"{pointer_angle:.1f}_{timestamp}")[0].boxes

        pointer_box = None
        max_y = -1  # max y height to see the lower bubble to target

        if target_box is not None:
            target_class = get_class_in_location(boxes, model, target_box)

        if (
            target_class != shooter_class
            or shooter_class is None
            or shooter_class != old_shooter_class
        ):
            shooter_class = get_shooter_class(boxes, model, SHOOTER_BOX)

        for i in range(len(boxes)):
            box = boxes.xyxy[i]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]

            if pointer_box is None and class_name == "pointer":
                pointer_box = box.cpu().numpy().astype(int)
                pointer_coords = (cx.item(), cy.item())
                continue

            if (
                target_angle is None
                and cy.item() < Y_THRESHOLD
                and shooter_class is not None
                and class_name.lower() == shooter_class.lower()
            ):
                if cy.item() > max_y:
                    max_y = cy.item()
                    target_coords = (cx.item(), cy.item())
                    target_box = box
                    print(f"target coords x, y: {target_coords}")
                continue

        if target_coords is not None:
            if pointer_box is not None:

                pointer_angle = get_angle(pointer_center, pointer_coords)
                print(f"Pointer angle: {pointer_angle:.2f}°")

                if target_angle is None:
                    target_angle = get_angle(pointer_center, target_coords)

                print(f"target angle: {target_angle:.2f}°")
                target_angle = aim_shoot(
                    pointer_angle,
                    target_angle,
                )

            else:
                print("pointer not found.")
        else:
            print("target not found.")
        time.sleep(1)
main()
