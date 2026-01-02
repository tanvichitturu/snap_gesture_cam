import cv2
import mediapipe as mp 
import numpy as np
import csv
import time
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
prev_time = 0
alpha = 0.2
smoothed_x = None
prev_x = None
swipe_threshold = 0.08
cooldown_time = 0.7
last_swipe_time =0
filters = ["Normal","B&W","Sepia","Cyberpunk","Warm"]
current_filter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = (cv2.flip(frame,1))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            x = wrist.x 
            if smoothed_x is None:
                smoothed_x = x
            else:
                smoothed_x = alpha * x + (1 - alpha) * smoothed_x
            h , w, _ = frame.shape
            cx = int(smoothed_x*w)
            cy = int(wrist.y*h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
            current_time = time.time()
            if prev_x is not None:
                dx = smoothed_x - prev_x
                if abs(dx) > swipe_threshold and (current_time - last_swipe_time) > cooldown_time:
                    if dx > 0:
                        current_filter = (current_filter+1)%len(filters)
                    else:
                        current_filter = (current_filter-1)%len(filters)
                        print("Filter:",filters[current_filter])
                    last_swipe_time = current_time
            prev_x = smoothed_x
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}",(10,40),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.putText(frame, f"Filter: {filters[current_filter]}",(10,80),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow("webcam",frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()