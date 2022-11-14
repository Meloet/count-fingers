import cv2
import mediapipe as mp
camara=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
tipid=[8,12,16,20]
def drawHandLandmarks(image,hands_landmarks):
    if hands_landmarks:
        for lm in hands_landmarks:
            mp_drawing.draw_landmarks(image,lm,mp_hands.HAND_CONNECTIONS)
def countFingers(image,hands_landmarks):
    if hands_landmarks:
        landmarks=hands_landmarks[0].landmark
        fingers=[]
        for lm_index in tipid:
            fingertipY=landmarks[lm_index].y
            fingerBottomY=landmarks[lm_index-2].y
            if fingertipY < fingerBottomY:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers=fingers.count(1)
        text=f'Fingers:{totalFingers}'
        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
while True:
    success,image=camara.read()
    image=cv2.flip(image,1)
    results=hands.process(image)
    hands_landmarks=results.multi_hand_landmarks
    drawHandLandmarks(image,hands_landmarks)
    countFingers(image,hands_landmarks)
    cv2.imshow("Media Controller",image)
    if cv2.waitKey(1)==32:
        break