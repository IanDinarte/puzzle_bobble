import pyautogui
import time
import os
import torch
from PIL import Image


def run_bot(decision):
    if "blueBubble" in decision:
        pyautogui.press(" ")
    elif "redBubble" in decision:
        pyautogui.press(" ")
    elif "purpleBubble" in decision:
        pyautogui.press(" ")
    elif "whiteBubble" in decision:
        pyautogui.press(" ")
    elif "grayBubble" in decision:
        pyautogui.press(" ")


def take_screenshot(model):
    screenx_center = 1920 / 2
    screeny_center = 1080 / 2
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    while True:

        decision = {
            "blueBubble": False,
            "redBubble": False,
            "whiteBubble": False,
            "grayBubble": False,
            "purpleBubble": False,
            "yellowBubble": False,
            "greenBubble": False,
            "orangeBubble": False,
        }

        #detect what bubble will be shot
        target_box = torch.tensor([924.0000, 860.0000, 994.000, 942.000])
        def point_in_box(cx, cy, box):
            x1, y1, x2, y2 = box
            return x1 <= cx <= x2 and y1 <= cy <= y2
        
        screenshot = pyautogui.screenshot()
        screenshot = Image.frombytes('RGB', screenshot.size, screenshot.tobytes())

        results = model([screenshot], conf=.70)

        boxes = results[0].boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i]  # (x1, y1, x2, y2)
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            class_id = int(boxes.cls[i].item())

            if point_in_box(cx.item(), cy.item(), target_box):
                print(f"{class_names[class_id]} is in the target location at center ({cx.item():.1f}, {cy.item():.1f})")
        

        result = model([screenshot], conf=.70)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best.pt")

    screenshot_thread = threading.Thread(
        target=take_screenshot, args=(stop_event, model)
    )
    screenshot_thread.start()

    model = YOLO(model_path)
