import cv2
import numpy as np


cap = cv2.VideoCapture(0) #load the camera (ex: pi cam or webcam)
#Open your camera (process)
while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #conversion of colorful into HSV

    #red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    #make a mask of red
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    #Finding counters of our red masked object
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
#now Draw rectangle and center line on our object
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x_medium = int((x + x + w) / 2)
        cv2.line(frame, (x_medium, 0), (x_medium, 720), (0, 255, 0), 2)
        break


    cv2.imshow("Frame", frame)
    cv2.imshow("mask", red_mask)
    key = cv2.waitKey(1)

    if key == 27:   #27 is equal to the Esc key on keyboard
        break

cap.release()
cv2.destroyAllWindows()