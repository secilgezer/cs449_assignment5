import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import os

# Suppress Mediapipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MediaPipe settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Gesture history
position_history = deque(maxlen=10)  # Keep the last 10 hand positions

# Tkinter window
root = tk.Tk()
root.title("Gesture Menu")

# Main container
main_container = tk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=True)

# Left panel (camera feed)
left_panel = tk.Frame(main_container)
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Canvas for the camera feed
canvas = tk.Canvas(left_panel, bg='black')
canvas.pack(fill=tk.BOTH, expand=True)

# Right panel (menu)
right_panel = tk.Frame(main_container, bg='#1E1E1E', width=250)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

# Menu title
menu_label = tk.Label(right_panel, text="Menu", font=('Arial', 14, 'bold'), bg='#1E1E1E', fg='white')
menu_label.pack(pady=10)

# Grid frame for menu items
menu_frame = tk.Frame(right_panel, bg='#1E1E1E')
menu_frame.pack(fill=tk.X, pady=10)

# Menu items
menu_items = []
menu_texts = [
    ["Item 1", "Item 2"],
    ["Item 3", "Item 4"],
    ["Item 5", "Item 6"],
    ["Item 7", "Item 8"],
    ["Item 9", "Item 10"],
    ["Item 11", "Item 12"]
]
current_row = 0
current_col = 0

# Place menu items in a grid
for i, row in enumerate(menu_texts):
    row_items = []
    for j, text in enumerate(row):
        btn = tk.Label(menu_frame, text=text, font=('Arial', 12),
                      bg='#2C2C2C', fg='white', pady=10, padx=20)
        btn.grid(row=i, column=j, padx=2, pady=2, sticky='ew')
        row_items.append(btn)
    menu_items.append(row_items)

# Make columns equally wide
menu_frame.grid_columnconfigure(0, weight=1)
menu_frame.grid_columnconfigure(1, weight=1)

# Gesture information
gesture_label = tk.Label(right_panel, text="No Gesture", font=('Arial', 12),
                        bg='#1E1E1E', fg='white', wraplength=180)
gesture_label.pack(pady=20)

# Cooldown variables
last_selection_time = 0
selection_cooldown = 1.0  # 1 second cooldown for selecting an item

# Highlight selected menu item
def highlight_item(row, col):
    """Highlight the selected menu item"""
    for i, row_items in enumerate(menu_items):
        for j, item in enumerate(row_items):
            if i == row and j == col:
                item.configure(bg='#0078D4')
            else:
                item.configure(bg='#2C2C2C')

# Select a menu item
def select_item(row, col):
    """Perform the click action on the selected menu item and provide visual feedback."""
    selected_item = menu_items[row][col]
    print(f"Selected: {selected_item.cget('text')}")

    # Highlight the clicked item in green
    selected_item.configure(bg='green')
    root.after(900, lambda: selected_item.configure(bg='#0078D4'))  # Revert to the original color after 900ms

    # Display "Clicked" feedback
    feedback_label = tk.Label(root, text="Clicked!", font=('Arial', 20, 'bold'),
                              bg='black', fg='white')
    feedback_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center of the window

    # Remove the feedback after 1 second
    root.after(1000, feedback_label.destroy)

# Gesture detection functions
def detect_swipe_left(history):
    if len(history) < 2:
        return False
    x1, _ = history[0]
    x2, _ = history[-1]
    return x2 < x1 - 0.2

def detect_swipe_right(history):
    if len(history) < 2:
        return False
    x1, _ = history[0]
    x2, _ = history[-1]
    return x2 > x1 + 0.2

def detect_swipe_up(history):
    if len(history) < 2:
        return False
    _, y1 = history[0]
    _, y2 = history[-1]
    return y2 < y1 - 0.2

def detect_swipe_down(history):
    if len(history) < 2:
        return False
    _, y1 = history[0]
    _, y2 = history[-1]
    return y2 > y1 + 0.2

def detect_thumbs_up(landmarks):
    """Detect Thumbs Up gesture"""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    is_thumb_up = thumb_tip.y < thumb_ip.y
    are_other_fingers_down = (
            index_tip.y > landmarks[6].y and
            middle_tip.y > landmarks[10].y and
            ring_tip.y > landmarks[14].y and
            pinky_tip.y > landmarks[18].y
    )
    return is_thumb_up and are_other_fingers_down

def detect_open_palm(landmarks):
    """Check if the hand is fully open."""
    fingers_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    fingers_mcp = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    return all(tip.y < mcp.y for tip, mcp in zip(fingers_tips, fingers_mcp))

# Video capture settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_skip = 3  # Process every 3rd frame
frame_count = 0

# Gesture detection threading
gesture_thread_lock = threading.Lock()
gesture_results = None
last_detection_time = 0
detection_interval = 0.2  # Process gestures every 200ms

def process_gesture(image_rgb):
    """Process gesture in a separate thread"""
    global gesture_results
    results = hands.process(image_rgb)
    with gesture_thread_lock:
        gesture_results = results

def update_frame():
    """Update the camera feed and detect gestures"""
    global current_row, current_col, frame_count, gesture_results, last_detection_time, last_selection_time
    success, image = cap.read()
    if success:
        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames to reduce lag
            root.after(10, update_frame)
            return

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Start gesture detection in a separate thread every 200ms
        if time.time() - last_detection_time > detection_interval:
            threading.Thread(target=process_gesture, args=(image_rgb,)).start()
            last_detection_time = time.time()

        # Draw results if available
        with gesture_thread_lock:
            if gesture_results and gesture_results.multi_hand_landmarks:
                hand_landmarks = gesture_results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                index_finger = landmarks[8]
                position_history.append((index_finger.x, index_finger.y))

                gesture_detected = "No Gesture"

                # Handle "Thumbs Up" gesture with cooldown
                if detect_thumbs_up(landmarks):
                    current_time = time.time()
                    if current_time - last_selection_time > selection_cooldown:
                        gesture_detected = "Thumbs Up"
                        select_item(current_row, current_col)  # Trigger item selection
                        last_selection_time = current_time  # Reset cooldown timer
                elif detect_open_palm(landmarks):
                    if detect_swipe_left(position_history):
                        gesture_detected = "Swipe Left"
                        current_col = max(0, current_col - 1)
                        highlight_item(current_row, current_col)
                    elif detect_swipe_right(position_history):
                        gesture_detected = "Swipe Right"
                        current_col = min(len(menu_items[0]) - 1, current_col + 1)
                        highlight_item(current_row, current_col)
                    elif detect_swipe_up(position_history):
                        gesture_detected = "Swipe Up"
                        current_row = max(0, current_row - 1)
                        highlight_item(current_row, current_col)
                    elif detect_swipe_down(position_history):
                        gesture_detected = "Swipe Down"
                        current_row = min(len(menu_items) - 1, current_row + 1)
                        highlight_item(current_row, current_col)
                    else:
                        gesture_detected = "Open Palm"

                gesture_label.config(text=f"Detected: {gesture_detected}")
                cv2.putText(image, f'Gesture: {gesture_detected}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update canvas
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width > 0 and canvas_height > 0:
            image = image.resize((canvas_width, canvas_height))
            photo = ImageTk.PhotoImage(image=image)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.photo = photo

    root.after(50, update_frame)  # Update every 50ms

# Highlight the first item and start the update loop
highlight_item(current_row, current_col)
update_frame()

# Cleanup on window close
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the main loop
root.mainloop()
