import cv2
import mediapipe as mp


# MediaPipe el modülü
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Başparmağı kontrol et
            thumb_tip = hand_landmarks.landmark[4]  # Başparmak ucu
            thumb_base = hand_landmarks.landmark[2]  # Başparmak eklem noktası

            # Başparmak yukarıdaysa
            if thumb_tip.y < thumb_base.y:
                cv2.putText(img, "Thumbs Up Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
