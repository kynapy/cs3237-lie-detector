import cv2
import os

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while cap.isOpened():
    break
    success, img = cap.read()
    if success:
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

path = "images"
if not os.path.exists(path):
    os.makedirs(path)
    print("Directory created")