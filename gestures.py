import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task') #loads pretrained handtracking model
options = vision.GestureRecognizerOptions(base_options=base_options) #configurations for the model
recognizer = vision.GestureRecognizer.create_from_options(options) #initialize hand detector using the configurations
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
    frame = cv2.flip(frame, 1) #flip the image horizontally
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB) #convert open cv image to mp image
    recognizer_result = recognizer.recognize(mpimage)
    h, w, c = frame.shape

    if recognizer_result.gestures !=[]: #if any gesture is recognized...
        top_gesture = recognizer_result.gestures[0][0] 
        hand_landmarks = recognizer_result.hand_landmarks
        resultImage = (top_gesture, hand_landmarks)
        points = []

        #draw landmarks
        for lm in resultImage[1][0]:
            print(lm)
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x,y))
            cv2.circle(frame, (x, y), 5, (0,255,255), -1)
        #draw connections
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (255,0,0), 2)

        #detect different gestures & add a caption
        if resultImage[0].category_name == "Thumb_Up":
            frame = cv2.putText(frame, "Thumbs up!", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
       
        elif resultImage[0].category_name == "Thumb_Down":
            frame = cv2.putText(frame, "Thumbs Down!", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
       
        elif resultImage[0].category_name == "Open_Palm":
            frame = cv2.putText(frame, "Stop", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
   
    cv2.imshow("cam feed", frame)
    cv2.waitKey(1)

