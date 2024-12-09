import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time  # Import to keep track of gesture duration

# MediaPipe settings for hand tracking
mp_hands = mp.solutions.hands  # Initialize MediaPipe Hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Set detection and tracking confidence
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing hand landmarks

# History of hand positions (for swipe detection)
position_history = deque(maxlen=10)  # Store the last 10 positions of the hand

# Variables for gesture duration control
last_gesture_time = 0  # Time of the last detected gesture
last_gesture = ""  # Name of the last detected gesture
gesture_duration = 1.0  # How long a gesture should be displayed (in seconds)

# Functions to detect gestures
def detect_swipe_left(history):
    """Detects if the hand swiped left."""
    if len(history) < 2:
        return False
    x1, _ = history[0]  # X-coordinate of the initial position
    x2, _ = history[-1]  # X-coordinate of the last position
    return x2 < x1 - 0.2  # Check if there was a significant movement to the left


def detect_swipe_right(history):
    """Detects if the hand swiped right."""
    if len(history) < 2:
        return False
    x1, _ = history[0]  # X-coordinate of the initial position
    x2, _ = history[-1]  # X-coordinate of the last position
    return x2 > x1 + 0.2  # Check if there was a significant movement to the right


def detect_swipe_up(history):
    """Detects if the hand swiped up."""
    if len(history) < 2:
        return False
    _, y1 = history[0]  # Y-coordinate of the initial position
    _, y2 = history[-1]  # Y-coordinate of the last position
    return y2 < y1 - 0.2  # Check if there was a significant movement upwards


def detect_swipe_down(history):
    """Detects if the hand swiped down."""
    if len(history) < 2:
        return False
    _, y1 = history[0]  # Y-coordinate of the initial position
    _, y2 = history[-1]  # Y-coordinate of the last position
    return y2 > y1 + 0.2  # Check if there was a significant movement downwards


def detect_thumbs_up(landmarks):
    """Detects the thumbs-up gesture (only the thumb is up, the rest of the fingers are down)."""
    thumb_tip = landmarks[4]  # Thumb tip position
    thumb_ip = landmarks[3]  # Thumb interphalangeal joint position
    index_tip = landmarks[8]  # Index finger tip
    middle_tip = landmarks[12]  # Middle finger tip
    ring_tip = landmarks[16]  # Ring finger tip
    pinky_tip = landmarks[20]  # Pinky tip

    # Check if the thumb is up (tip higher than IP joint) and the other fingers are down
    is_thumb_up = thumb_tip.y < thumb_ip.y  # Check if thumb is up
    are_other_fingers_down = (
        index_tip.y > landmarks[6].y and  # Check if index finger is down
        middle_tip.y > landmarks[10].y and  # Check if middle finger is down
        ring_tip.y > landmarks[14].y and  # Check if ring finger is down
        pinky_tip.y > landmarks[18].y  # Check if pinky finger is down
    )
    return is_thumb_up and are_other_fingers_down  # Return True if thumb is up and others are down


def detect_open_palm(landmarks):
    """Detects if the hand is open (all fingers extended)."""
    fingers_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]  # Tips of the 4 fingers
    fingers_mcp = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]  # MCP (base) points of the fingers
    # Check if all the tips of the fingers are above the MCP (base) points
    return all(tip.y < mcp.y for tip, mcp in zip(fingers_tips, fingers_mcp))  


# Start video capture
cap = cv2.VideoCapture(0)  # Open the webcam (default camera 0)

while cap.isOpened():
    success, image = cap.read()  # Capture a frame from the webcam
    if not success:
        print("Camera frame is empty.")
        continue

    image = cv2.flip(image, 1)  # Flip the image horizontally (like a mirror)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB (for MediaPipe)
    results = hands.process(image_rgb)  # Process the frame to detect hands

    current_time = time.time()  # Get the current time

    if results.multi_hand_landmarks:  # If hands are detected in the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw landmarks on the hand

            landmarks = hand_landmarks.landmark  # Extract hand landmarks
            index_finger = landmarks[8]  # Get the position of the index finger
            position_history.append((index_finger.x, index_finger.y))  # Update position history

            gesture_detected = "No Gesture"  # Default gesture

            # Check for various gestures
            if detect_thumbs_up(landmarks):
                gesture_detected = "Thumbs Up"
            elif detect_open_palm(landmarks):
                if detect_swipe_left(position_history):
                    if last_gesture != "Swipe Left" or (current_time - last_gesture_time) > gesture_duration:
                        last_gesture = "Swipe Left"
                        last_gesture_time = current_time
                    gesture_detected = last_gesture
                elif detect_swipe_right(position_history):
                    if last_gesture != "Swipe Right" or (current_time - last_gesture_time) > gesture_duration:
                        last_gesture = "Swipe Right"
                        last_gesture_time = current_time
                    gesture_detected = last_gesture
                elif detect_swipe_up(position_history):
                    if last_gesture != "Swipe Up" or (current_time - last_gesture_time) > gesture_duration:
                        last_gesture = "Swipe Up"
                        last_gesture_time = current_time
                    gesture_detected = last_gesture
                elif detect_swipe_down(position_history):
                    if last_gesture != "Swipe Down" or (current_time - last_gesture_time) > gesture_duration:
                        last_gesture = "Swipe Down"
                        last_gesture_time = current_time
                    gesture_detected = last_gesture
                elif (current_time - last_gesture_time) > gesture_duration:
                    gesture_detected = "Open Palm"
                else:
                    gesture_detected = last_gesture

            # Display the gesture name on the screen
            cv2.putText(image, f'Gesture: {gesture_detected}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display "Hand Detected" if a hand is detected
    if results.multi_hand_landmarks:
        cv2.putText(image, "Hand Detected", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the image in a window
    cv2.imshow('Hand Gesture Recognition', image)

    # Exit the loop when the ESC key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()

