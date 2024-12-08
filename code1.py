import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk

# MediaPipe ayarları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Hareket geçmişi
position_history = deque(maxlen=10)  # Son 10 el konumunu tutar

# Tkinter penceresi
root = tk.Tk()
root.title("Gesture Menu")

# Ana container
main_container = tk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=True)

# Sol panel (kamera görüntüsü için)
left_panel = tk.Frame(main_container)
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Kamera görüntüsü için canvas
canvas = tk.Canvas(left_panel, bg='black')
canvas.pack(fill=tk.BOTH, expand=True)

# Sağ panel (menü için)
right_panel = tk.Frame(main_container, bg='#1E1E1E', width=250)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

# Menü başlığı
menu_label = tk.Label(right_panel, text="Menu", font=('Arial', 14, 'bold'), bg='#1E1E1E', fg='white')
menu_label.pack(pady=10)

# Menü öğeleri için grid frame
menu_frame = tk.Frame(right_panel, bg='#1E1E1E')
menu_frame.pack(fill=tk.X, pady=10)

# Menü öğeleri
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

# Menü öğelerini grid olarak yerleştir
for i, row in enumerate(menu_texts):
    row_items = []
    for j, text in enumerate(row):
        btn = tk.Label(menu_frame, text=text, font=('Arial', 12),
                      bg='#2C2C2C', fg='white', pady=10, padx=20)
        btn.grid(row=i, column=j, padx=2, pady=2, sticky='ew')
        row_items.append(btn)
    menu_items.append(row_items)

# Sütunları eşit genişlikte yap
menu_frame.grid_columnconfigure(0, weight=1)
menu_frame.grid_columnconfigure(1, weight=1)

# Gesture bilgisi
gesture_label = tk.Label(right_panel, text="No Gesture", font=('Arial', 12), 
                        bg='#1E1E1E', fg='white', wraplength=180)
gesture_label.pack(pady=20)

def highlight_item(row, col):
    """Menü öğesini vurgula"""
    for i, row_items in enumerate(menu_items):
        for j, item in enumerate(row_items):
            if i == row and j == col:
                item.configure(bg='#0078D4')
            else:
                item.configure(bg='#2C2C2C')

def detect_swipe_left(history):
    """Sola kaydırma hareketi algıla"""
    if len(history) < 2:
        return False
    x1, _ = history[0]
    x2, _ = history[-1]
    return x2 < x1 - 0.2

def detect_swipe_right(history):
    """Sağa kaydırma hareketi algıla"""
    if len(history) < 2:
        return False
    x1, _ = history[0]
    x2, _ = history[-1]
    return x2 > x1 + 0.2

def detect_swipe_up(history):
    """Yukarı kaydırma hareketi algıla"""
    if len(history) < 2:
        return False
    _, y1 = history[0]
    _, y2 = history[-1]
    return y2 < y1 - 0.2

def detect_swipe_down(history):
    """Aşağı kaydırma hareketi algıla"""
    if len(history) < 2:
        return False
    _, y1 = history[0]
    _, y2 = history[-1]
    return y2 > y1 + 0.2

def detect_thumbs_up(landmarks):
    """Thumbs Up jestini algıla"""
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
    """Elin tamamen açık olup olmadığını kontrol et"""
    fingers_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    fingers_mcp = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    return all(tip.y < mcp.y for tip, mcp in zip(fingers_tips, fingers_mcp))

def update_frame():
    """Kamera görüntüsünü güncelle ve gesture'ları algıla"""
    global current_row, current_col
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
                elif detect_open_palm(landmarks):
                    if detect_swipe_left(position_history):
                        gesture_detected = "Swipe Left"
                        # Sola git
                        current_col = max(0, current_col - 1)
                        highlight_item(current_row, current_col)
                    elif detect_swipe_right(position_history):
                        gesture_detected = "Swipe Right"
                        # Sağa git
                        current_col = min(len(menu_items[0]) - 1, current_col + 1)
                        highlight_item(current_row, current_col)
                    elif detect_swipe_up(position_history):
                        gesture_detected = "Swipe Up"
                        # Yukarı git
                        current_row = max(0, current_row - 1)
                        highlight_item(current_row, current_col)
                    elif detect_swipe_down(position_history):
                        gesture_detected = "Swipe Down"
                        # Aşağı git
                        current_row = min(len(menu_items) - 1, current_row + 1)
                        highlight_item(current_row, current_col)
                    else:
                        gesture_detected = "Open Palm"

                # Gesture bilgisini güncelle
                gesture_label.config(text=f"Detected: {gesture_detected}")
                cv2.putText(image, f'Gesture: {gesture_detected}', (10, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Görüntüyü canvas'a yerleştir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width > 0 and canvas_height > 0:
            image = image.resize((canvas_width, canvas_height))
            photo = ImageTk.PhotoImage(image=image)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.photo = photo

    root.after(10, update_frame)

# Video yakalama başlat
cap = cv2.VideoCapture(0)

# Pencere boyutu
root.geometry("1200x800")

# İlk frame'i göster ve ilk öğeyi vurgula
highlight_item(current_row, current_col)
update_frame()

# Pencere kapandığında temizlik
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Ana döngüyü başlat
root.mainloop()
