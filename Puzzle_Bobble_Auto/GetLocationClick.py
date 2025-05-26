from pynput import mouse

# gets the coordinates of the location clicked
def on_click(x, y, button, pressed):
    if pressed:
        print(f"Mouse clicked at ({x}, {y})")

# listens to the mouse click
with mouse.Listener(on_click=on_click) as listener:
    listener.join()