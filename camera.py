import cv2
import os
from time import sleep

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

path = "images"
if not os.path.exists(path):
    os.makedirs(path)
    print("Directory created")

os.chdir(path)

count = 0
while cap.isOpened():
    filename = "image_" + str(count) + ".png"
    count += 1
    success, img = cap.read()
    if success:
        #cv2.imshow("Result", img)
        cv2.imwrite(filename, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    sleep(10)