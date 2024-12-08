import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# MediaPipe ayarları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Hareket geçmişi
position_history = deque(maxlen=10)  # Son 10 el konumunu tutar


# Gesture kontrol fonksiyonları
def detect_swipe_left(history):
    if len(history) < 2:
        return False
    x1, _ = history[0]
    x2, _ = history[-1]
    return x2 < x1 - 0.2  # Sağdan sola hareketi algıla


def detect_swipe_right(history):
    if len(history) < 2:
        return False
    x1, _ = history[0]
    x2, _ = history[-1]
    return x2 > x1 + 0.2  # Soldan sağa hareketi algıla


def detect_swipe_up(history):
    if len(history) < 2:
        return False
    _, y1 = history[0]
    _, y2 = history[-1]
    return y2 < y1 - 0.2  # Aşağıdan yukarıya hareketi algıla


def detect_swipe_down(history):
    if len(history) < 2:
        return False
    _, y1 = history[0]
    _, y2 = history[-1]
    return y2 > y1 + 0.2  # Yukarıdan aşağıya hareketi algıla


def detect_thumbs_up(landmarks):
    """Thumbs Up jestini algıla (sadece başparmak yukarıda, diğer parmaklar aşağıda)"""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Thumb up kontrolü: Başparmak yukarıda, diğer parmaklar aşağıda
    is_thumb_up = thumb_tip.y < thumb_ip.y  # Başparmak yukarıda
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
    return all(tip.y < mcp.y for tip, mcp in zip(fingers_tips, fingers_mcp))  # Parmakların açık olması


# Video başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Kamera çerçevesi boş.")
        continue

    image = cv2.flip(image, 1)  # Görüntüyü yatay çevir (ayna efekti)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8]  # İşaret parmağı
            position_history.append((index_finger.x, index_finger.y))  # El konumunu güncelle

            gesture_detected = "No Gesture"

            if detect_thumbs_up(landmarks):
                gesture_detected = "Thumbs Up"
            elif detect_open_palm(landmarks):
                if detect_swipe_left(position_history):
                    gesture_detected = "Swipe Left"
                elif detect_swipe_right(position_history):
                    gesture_detected = "Swipe Right"
                elif detect_swipe_up(position_history):
                    gesture_detected = "Swipe Up"
                elif detect_swipe_down(position_history):
                    gesture_detected = "Swipe Down"
                else:
                    gesture_detected = "Open Palm"

            # Jest adını ekrana yazdır
            cv2.putText(image, f'Gesture: {gesture_detected}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Görüntüyü göster
    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC tuşuna basıldığında çık
        break

cap.release()
cv2.destroyAllWindows()
