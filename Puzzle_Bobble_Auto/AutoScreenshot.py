import pyautogui
import time
import os


if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

while True:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Taking screenshot...")
    screenshot = pyautogui.screenshot()
    screenshot.save(f"screenshots/screenshot_{timestamp}.png")
    time.sleep(5)
    