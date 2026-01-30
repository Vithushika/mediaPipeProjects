import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task') #loads pretrained handtracking model
options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2) #configurations for the model + how many hands to detect
detector = vision.HandLandmarker.create_from_options(options) #initialize hand detector using configurations

cap = cv2.VideoCapture(0)

connections = [(0,1), (1,2), (2,3), (3,4), #thumb
               (0,5), (5,6), (6,7), (7,8), #index
               (5,9), (9,10), (10,11), (11,12), #middle
               (9,13), (13,14), (14,15), (15,16), #ring
               (13,17), (17,18), (18,19), (19,20), #pinky
               (0,17) #wrist to pinky
               ]

while True:
    ref, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB) #convert open cv image to mp image
    detection_result = detector.detect(mpimage) #take each frame and detect hand landmarks
    h, w, c = frame.shape
    if detection_result.hand_landmarks: #if hand detected...
        for hand in detection_result.hand_landmarks: #for each hand
            points = []

            #drawing landmarks
            for lm in hand: #for each landmark on that hand
                cx, cy = int(lm.x*w), int(lm.y*h)
                points.append((cx,cy))
                cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)

            #drawing connections
            for start, end in connections:
                cv2.line(frame, points[start], points[end], (255,0,0), 2)
   
    cv2.imshow("cam feed", frame)
    cv2.waitKey(1)
