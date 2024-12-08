import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import ttk
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
right_panel = tk.Frame(main_container, bg='#1E1E1E', width=300)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

# Menu title
menu_label = tk.Label(right_panel, text="Menu", font=('Arial', 14, 'bold'), bg='#1E1E1E', fg='white')
menu_label.pack(pady=10)

# Scrollable canvas for menu items
menu_canvas = tk.Canvas(right_panel, bg='#1E1E1E')
menu_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Scrollbars
v_scrollbar = tk.Scrollbar(right_panel, orient=tk.VERTICAL, command=menu_canvas.yview)
v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
h_scrollbar = tk.Scrollbar(right_panel, orient=tk.HORIZONTAL, command=menu_canvas.xview)
h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

menu_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

# Frame inside the canvas
menu_inner_frame = tk.Frame(menu_canvas, bg='#1E1E1E')

# Function to update scroll region
def on_frame_configure(event):
    menu_canvas.configure(scrollregion=menu_canvas.bbox("all"))

menu_inner_frame.bind("<Configure>", on_frame_configure)

# Add the inner frame to the canvas
menu_canvas.create_window((0, 0), window=menu_inner_frame, anchor='nw')

# Menu items
menu_items = []
menu_texts = [
    ["Item 1", "Item 2"],
    ["Item 3", "Item 4"]
]
current_row = 0
current_col = 0

# Place menu items in a grid inside the inner frame
for i, row in enumerate(menu_texts):
    row_items = []
    for j, text in enumerate(row):
        btn = tk.Label(menu_inner_frame, text=text, font=('Arial', 24),  # Increased font size for demonstration
                      bg='#2C2C2C', fg='white', pady=50, padx=100)        # Increased padding for demonstration
        btn.grid(row=i, column=j, padx=2, pady=2, sticky='nsew')
        row_items.append(btn)
    menu_items.append(row_items)

# Make columns and rows equally wide and tall
for i in range(len(menu_texts)):
    menu_inner_frame.grid_rowconfigure(i, weight=1)
for j in range(len(menu_texts[0])):
    menu_inner_frame.grid_columnconfigure(j, weight=1)

# Gesture information
gesture_label = tk.Label(right_panel, text="No Gesture", font=('Arial', 12),
                        bg='#1E1E1E', fg='white', wraplength=180)
gesture_label.pack(pady=10)

# Variables to hold the last detected gesture
last_gesture = "No Gesture"
last_gesture_time = 0
gesture_display_duration = 1.0  # 1 second to display the gesture
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
    return x2 < x1 - 0.05  # Adjusted sensitivity

def detect_swipe_right(history):
    if len(history) < 2:
        return False
    x1, _ = history[0]
    x2, _ = history[-1]
    return x2 > x1 + 0.05  # Adjusted sensitivity

def detect_swipe_up(history):
    if len(history) < 2:
        return False
    _, y1 = history[0]
    _, y2 = history[-1]
    return y2 < y1 - 0.05  # Adjusted sensitivity

def detect_swipe_down(history):
    if len(history) < 2:
        return False
    _, y1 = history[0]
    _, y2 = history[-1]
    return y2 > y1 + 0.05  # Adjusted sensitivity

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

def update_frame():
    """Update the camera feed and detect gestures"""
    global current_row, current_col, last_gesture, last_gesture_time, last_selection_time
    success, image = cap.read()
    if success:
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                index_finger = landmarks[8]
                position_history.append((index_finger.x, index_finger.y))

                gesture_detected = "No Gesture"

                if detect_thumbs_up(landmarks):
                    gesture_detected = "Thumbs Up"
                    current_time = time.time()
                    if current_time - last_selection_time > selection_cooldown:
                        select_item(current_row, current_col)
                        last_selection_time = current_time
                elif detect_open_palm(landmarks):
                    if detect_swipe_left(position_history):
                        gesture_detected = "Swipe Left"
                        menu_canvas.xview_scroll(-1, "units")
                    elif detect_swipe_right(position_history):
                        gesture_detected = "Swipe Right"
                        menu_canvas.xview_scroll(1, "units")
                    elif detect_swipe_up(position_history):
                        gesture_detected = "Swipe Up"
                        menu_canvas.yview_scroll(-1, "units")
                    elif detect_swipe_down(position_history):
                        gesture_detected = "Swipe Down"
                        menu_canvas.yview_scroll(1, "units")
                    else:
                        gesture_detected = "Open Palm"

                if gesture_detected != "No Gesture":
                    last_gesture = gesture_detected
                    last_gesture_time = time.time()

        # Maintain gesture display for 1 second
        if time.time() - last_gesture_time < gesture_display_duration:
            gesture_label.config(text=f"Detected: {last_gesture}")
            cv2.putText(image, f'Gesture: {last_gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            gesture_label.config(text="No Gesture")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.photo = photo

    root.after(10, update_frame)

# Highlight the first item and start the update loop
highlight_item(current_row, current_col)
update_frame()

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
