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

# Gesture history to track the last 10 positions of the index finger
position_history = deque(maxlen=10)  # Keep the last 10 hand positions

# Tkinter window
root = tk.Tk()
root.attributes("-fullscreen", True) # Fullscreen mode
root.title("Gesture Control Interface") # Window title

# Main container to hold the left and right panels
main_container = tk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=True)

# Left panel (camera feed for hand detection)
left_panel = tk.Frame(main_container)
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Canvas for the camera feed
canvas = tk.Canvas(left_panel, bg='black')
canvas.pack(fill=tk.BOTH, expand=True)

# Right panel (menu items displayed like a PC screen)
right_panel = tk.Frame(main_container, bg='#1E1E1E', width=600)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

# Menu title
menu_label = tk.Label(right_panel, text="Menu", font=('Arial', 14, 'bold'), bg='#1E1E1E', fg='white')
menu_label.pack(pady=10)

# Create a frame for the canvas and scrollbar
menu_frame = tk.Frame(right_panel, bg='#1E1E1E')
menu_frame.pack(fill=tk.BOTH, expand=True, pady=10)

# Create a canvas within the menu frame
menu_canvas = tk.Canvas(menu_frame, bg='#1E1E1E')
menu_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar
menu_scrollbar = ttk.Scrollbar(menu_frame, orient=tk.VERTICAL, command=menu_canvas.yview)
menu_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas to use the scrollbar
menu_canvas.configure(yscrollcommand=menu_scrollbar.set)

# Inner frame to hold menu items
inner_menu_frame = tk.Frame(menu_canvas, bg='#1E1E1E')
menu_canvas.create_window((0, 0), window=inner_menu_frame, anchor='nw')

# Menu items
menu_items = []
menu_texts = [f"Item {i+1}" for i in range(30)]  # 30 items

# Place menu items in 2 columns
for row in range(0, len(menu_texts), 2):
    column_items = []
    for col in range(2):
        if row + col < len(menu_texts):
            btn = tk.Label(inner_menu_frame, text=menu_texts[row + col], font=('Arial', 12),
                           bg='#2C2C2C', fg='white', pady=10, padx=20)
            btn.grid(row=row//2, column=col, padx=2, pady=2, sticky='ew')
            column_items.append(btn)
    menu_items.append(column_items)

# Update scroll region
inner_menu_frame.update_idletasks()
menu_canvas.config(scrollregion=menu_canvas.bbox("all"))

# Make columns equally wide
inner_menu_frame.grid_columnconfigure(0, weight=1)
inner_menu_frame.grid_columnconfigure(1, weight=1)

# Gesture information
gesture_label = tk.Label(right_panel, text="No Gesture", font=('Arial', 12),
                         bg='#1E1E1E', fg='white', wraplength=180)
gesture_label.pack(pady=20)

# Pointer widget
pointer = tk.Label(right_panel, text="►", font=('Arial', 18), fg='red', bg='#1E1E1E')
pointer.place(x=0, y=0)

# Variables to hold the last detected gesture
last_gesture = "No Gesture"
last_gesture_time = 0
gesture_display_duration = 1.0  # 1 second to display the gesture
last_selection_time = 0
selection_cooldown = 1.0  # 1 second cooldown for selecting an item

# Current selection tracking
current_row = 0
current_col = 0

# Highlight selected menu item
def highlight_item(row, col):
    """Highlight the selected menu item"""
    for i, column in enumerate(menu_items):
        for j, item in enumerate(column):
            if i == row and j == col:
                item.configure(bg='#0078D4')

                # Scroll to the selected item
                item_y = item.winfo_y()
                menu_canvas.yview_moveto(item_y / inner_menu_frame.winfo_height())
            else:
                item.configure(bg='#2C2C2C')

# Select a menu item
def select_item(row, col):
    """Perform the click action on the selected menu item and provide visual feedback."""
    try:
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
    except IndexError:
        print("Invalid selection")

# Gesture detection functions (same as before)
def detect_pointing_up(landmarks):
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Ensure index finger is pointing up (tip higher than base)
    is_index_up = index_tip.y < index_mcp.y

    # Other fingers should be curled down
    other_fingers_down = (
        middle_tip.y > landmarks[9].y and
        ring_tip.y > landmarks[13].y and
        pinky_tip.y > landmarks[17].y
    )

    return is_index_up and other_fingers_down

def detect_pointing_down(landmarks):
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Ensure index finger is pointing down (tip lower than base)
    is_index_down = index_tip.y > index_mcp.y

    # Other fingers should be curled down
    other_fingers_down = (
        middle_tip.y > landmarks[9].y and
        ring_tip.y > landmarks[13].y and
        pinky_tip.y > landmarks[17].y
    )

    return is_index_down and other_fingers_down

def detect_swipe_left(history):
    """
    Detects if the user's hand has made a swipe-left gesture.
    Args:
        history (deque): A deque containing the x, y positions of the index finger over time.
    Returns:
        bool: True if a swipe-left gesture is detected, False otherwise.
    """
    if len(history) < 2:   # If there are fewer than 2 points in the history, a swipe cannot be detected
        return False
    x1, _ = history[0]  # Get the x-coordinate of the index finger from the first recorded position
    x2, _ = history[-1]   # Get the x-coordinate of the index finger from the last recorded position
    return x2 < x1 - 0.2    # Check if the finger has moved significantly to the left (x2 is much less than x1)
    # The threshold of 0.2 determines the minimum distance for a "swipe" to be registered

def detect_swipe_right(history):
     """
    Detects if the user's hand has made a swipe-right gesture.
    Args:
        history (deque): A deque containing the x, y positions of the index finger over time.
    Returns:
        bool: True if a swipe-right gesture is detected, False otherwise.
    """
    # If there are fewer than 2 points in the history, a swipe cannot be detected
    if len(history) < 2:
        return False
    # Get the x-coordinate of the index finger from the first recorded position
    x1, _ = history[0]
    # Get the x-coordinate of the index finger from the last recorded position
    x2, _ = history[-1]
    # Check if the finger has moved significantly to the right (x2 is much greater than x1)
    # The threshold of 0.2 determines the minimum distance for a "swipe" to be registered
    return x2 > x1 + 0.2

def detect_thumbs_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
   
    is_thumb_up = thumb_tip.y < thumb_ip.y # Check if the thumb is up and other fingers are down
    are_other_fingers_down = (
        index_tip.y > landmarks[6].y and
        middle_tip.y > landmarks[10].y and
        ring_tip.y > landmarks[14].y and
        pinky_tip.y > landmarks[18].y
    )
    return is_thumb_up and are_other_fingers_down

# Video capture settings
cap = cv2.VideoCapture(0)
# Function to update the position of the pointer widget on the menu canvas
def update_pointer_position(x, y):
    """Update the pointer position on the menu canvas"""
    pointer.place(x=x, y=y)
    item = root.winfo_containing(x + right_panel.winfo_x(), y + right_panel.winfo_y())
    if item and isinstance(item, tk.Label) and item.cget('bg') == '#0078D4':  # Ensure it's an item
        select_item(current_row, current_col)

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

                if detect_pointing_up(landmarks):
                    gesture_detected = "Pointing Up"
                    current_row = max(0, current_row - 1)
                    highlight_item(current_row, current_col)
                elif detect_pointing_down(landmarks):
                    gesture_detected = "Pointing Down"
                    current_row = min(len(menu_items) - 1, current_row + 1)
                    highlight_item(current_row, current_col)
                elif detect_thumbs_up(landmarks):
                    gesture_detected = "Thumbs Up"
                    current_time = time.time()
                    if current_time - last_selection_time > selection_cooldown:
                        select_item(current_row, current_col)
                        last_selection_time = current_time
                elif detect_swipe_left(position_history):
                    gesture_detected = "Swipe Left"
                    current_col = max(0, current_col - 1)
                    highlight_item(current_row, current_col)
                elif detect_swipe_right(position_history):
                    gesture_detected = "Swipe Right"
                    current_col = min(1, current_col + 1)
                    highlight_item(current_row, current_col)

                if gesture_detected != "No Gesture":
                    last_gesture = gesture_detected
                    last_gesture_time = time.time()

                # Update pointer position based on hand index finger position
                image_height, image_width, _ = image.shape
                pointer_x = int(index_finger.x * right_panel.winfo_width())
                pointer_y = int(index_finger.y * right_panel.winfo_height())
                update_pointer_position(pointer_x, pointer_y)
                
                # Check if pointer is over an item
                item = root.winfo_containing(pointer_x + right_panel.winfo_x(), pointer_y + right_panel.winfo_y())
                if item and isinstance(item, tk.Label) and item.cget('bg') == '#0078D4':
                    select_item(current_row, current_col)

        # Maintain gesture display for 1 second
        if time.time() - last_gesture_time < gesture_display_duration:
            gesture_label.config(text=f"Detected: {last_gesture}")
        else:
            gesture_label.config(text="No Gesture")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.photo = photo

    root.after(10, update_frame)

highlight_item(current_row, current_col)
update_frame()

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
