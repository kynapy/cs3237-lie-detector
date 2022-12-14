'''
Code used for collection of data for training.
'''


import cv2
import os
import serial
from time import sleep, strftime
from datetime import date, datetime

frameWidth = 480
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(10, 150)

path = "data/q35"
if not os.path.exists(path):
    os.makedirs(path)
    print("Directory created")
else:
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))

os.chdir(path)

serial = serial.Serial("/dev/cu.usbserial-10", 9600, timeout=1)
f = open("hrData.txt", "w")

currentTime = 0
count = 0
while cap.isOpened():
    now = datetime.now()
    if currentTime == now.strftime("%H:%M:%S"):
        count += 1
    else:
        currentTime = now.strftime("%H:%M:%S")
        count = 0
    filename = "image_" + currentTime + "_" + str(count) +".png"
    success, img = cap.read()
    if success:
        data = serial.readline()
        data = data.decode("utf-8", errors='ignore')
        f.write("(" + currentTime + "_" + str(count) + ") : ")
        f.write(data)
        print(data)
        imgShape = len(img)
        img = img[int(imgShape*0.2):int(imgShape*0.8), int(imgShape*0.2):int(imgShape*0.8)]
        img = cv2.resize(img, (256, 256), interpolation= cv2.INTER_CUBIC)
        cv2.imwrite(filename, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    sleep(0.2)