import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task') #loads pretrained handtracking model
options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2) #configurations for the model + how many hands to detect
detector = vision.HandLandmarker.create_from_options(options) #initialize hand detector using the configurations
cap = cv2.VideoCapture(0)
draw = False
drawings = []

connections = [(0,1), (1,2), (2,3), (3,4), #thumb
               (0,5), (5,6), (6,7), (7,8), #index
               (5,9), (9,10), (10,11), (11,12), #middle
               (9,13), (13,14), (14,15), (15,16), #ring
               (13,17), (17,18), (18,19), (19,20), #pinky
               (0,17) #wrist to pinky
               ]

while True:
    ref, frame = cap.read()
    if ref == False:
        continue
   
    #draw mode and esc to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break
    if key == ord(' '):
        cv2.waitKey(5) 
        draw = not draw
    if key == ord('c'):
        drawings = []
        cv2.waitKey(5)  
    
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB) #convert open cv image to mp image
    detection_result = detector.detect(mpimage) #take each frame and detect hand landmarks

    #drawing state
    if not draw:
        cv2.putText(frame, "Drawing: Off, Press Space to start", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if draw:
        cv2.putText(frame, "Drawing: On, Press Space to stop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if detection_result.hand_landmarks: #if hand detected...
        for handLms in detection_result.hand_landmarks: #for each landmark detected on hand
            points = []
            index = detection_result.hand_landmarks[0][8]
            ix, iy = int(index.x*w), int(index.y*h)
            cv2.circle(frame, (ix, iy), 8, (0,0,0), cv2.FILLED)
            
            #drawing landmarks
            for lm in handLms: #for each landmark on that hand
                x, y = int(lm.x*w), int(lm.y*h)
                points.append((x,y))
                cv2.circle(frame, (x, y), 5, (255,0,0), cv2.FILLED)
                
            #drawing connections
            for start, end in connections:
                cv2.line(frame, points[start], points[end], (255,0,0), 2)
   
            if draw:
                drawings.append((ix,iy))
       
    for point in drawings:
        cv2.circle(frame, point, 10, (255,255,255), -1)
       
    cv2.imshow("cam feed", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()