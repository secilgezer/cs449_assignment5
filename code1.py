import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
   return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_angle(p1, p2, p3):
   angle = math.degrees(math.atan2(p3.y - p2.y, p3.x - p2.x) - 
                       math.atan2(p1.y - p2.y, p1.x - p2.x))
   return abs(angle)

while True:
   success, img = cap.read()
   if not success:
       break
       
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   results = hands.process(img_rgb)

   if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
           mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
           
           # Tüm parmak noktaları
           thumb_tip = hand_landmarks.landmark[4]
           thumb_ip = hand_landmarks.landmark[3]
           thumb_mcp = hand_landmarks.landmark[2]
           
           index_tip = hand_landmarks.landmark[8]
           index_pip = hand_landmarks.landmark[7]
           index_mcp = hand_landmarks.landmark[5]
           
           middle_tip = hand_landmarks.landmark[12]
           middle_pip = hand_landmarks.landmark[11]
           middle_mcp = hand_landmarks.landmark[9]
           
           ring_tip = hand_landmarks.landmark[16]
           ring_pip = hand_landmarks.landmark[15]
           ring_mcp = hand_landmarks.landmark[13]
           
           pinky_tip = hand_landmarks.landmark[20]
           pinky_pip = hand_landmarks.landmark[19]
           pinky_mcp = hand_landmarks.landmark[17]
           
           wrist = hand_landmarks.landmark[0]

           # 1. Tek tek parmak yukarı/aşağı kontrolleri
           if thumb_tip.y < thumb_mcp.y:
               cv2.putText(img, "Thumb Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
           
           if index_tip.y < index_mcp.y:
               cv2.putText(img, "Index Up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
               
           if middle_tip.y < middle_mcp.y:
               cv2.putText(img, "Middle Up", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
               
           if ring_tip.y < ring_mcp.y:
               cv2.putText(img, "Ring Up", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
               
           if pinky_tip.y < pinky_mcp.y:
               cv2.putText(img, "Pinky Up", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

           # 2. El pozisyonları
           # Yumruk
           if all(calculate_distance(tip, wrist) < 0.3 for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
               cv2.putText(img, "Fist", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
           
           # Düz el
           if all(abs(tip.z - wrist.z) < 0.1 for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
               cv2.putText(img, "Flat Hand", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

           # 3. Parmak kombinasyonları
           # Zafer işareti
           if (index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y and 
               ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y):
               cv2.putText(img, "Victory", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

           # OK işareti
           if calculate_distance(thumb_tip, index_tip) < 0.05:
               cv2.putText(img, "OK Sign", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

           # Rock işareti
           if (index_tip.y > index_mcp.y and pinky_tip.y < pinky_mcp.y and 
               middle_tip.y > middle_mcp.y and ring_tip.y > ring_mcp.y):
               cv2.putText(img, "Rock Sign", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

           # 4. Dinamik hareketler
           # El sallama
           if all(tip.x > wrist.x + 0.2 for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
               cv2.putText(img, "Waving", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

           # Tutma hareketi
           if all(calculate_distance(tip, thumb_tip) < 0.1 for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
               cv2.putText(img, "Grab", (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

           # İşaret etme
           if (index_tip.y < index_mcp.y and 
               all(tip.y > mcp.y for tip, mcp in [(middle_tip, middle_mcp), 
                                                 (ring_tip, ring_mcp), 
                                                 (pinky_tip, pinky_mcp)])):
               cv2.putText(img, "Pointing", (550, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

           # Silah işareti
           if (index_tip.y < index_mcp.y and thumb_tip.y < thumb_mcp.y and
               all(tip.y > mcp.y for tip, mcp in [(middle_tip, middle_mcp),
                                                 (ring_tip, ring_mcp),
                                                 (pinky_tip, pinky_mcp)])):
               cv2.putText(img, "Gun Sign", (550, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

           # Telefon işareti
           if (pinky_tip.y < pinky_mcp.y and thumb_tip.y < thumb_mcp.y and
               all(tip.y > mcp.y for tip, mcp in [(index_tip, index_mcp),
                                                 (middle_tip, middle_mcp),
                                                 (ring_tip, ring_mcp)])):
               cv2.putText(img, "Phone", (550, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

           # 5. Parmak açıları
           for finger_name, (tip, pip, mcp) in {
               "Thumb": (thumb_tip, thumb_ip, thumb_mcp),
               "Index": (index_tip, index_pip, index_mcp),
               "Middle": (middle_tip, middle_pip, middle_mcp),
               "Ring": (ring_tip, ring_pip, ring_mcp),
               "Pinky": (pinky_tip, pinky_pip, pinky_mcp)
           }.items():
               angle = calculate_angle(tip, pip, mcp)
               cv2.putText(img, f"{finger_name} Angle: {int(angle)}", 
                          (50, 300 + 50 * list({"Thumb", "Index", "Middle", "Ring", "Pinky"}).index(finger_name)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

   cv2.imshow("Hand Gesture", img)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()
