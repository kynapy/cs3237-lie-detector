'''
This code is used for the camera to upload the images to CS3237/Group_22/data/images channel

'''

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from time import sleep
import json
import serial
from datetime import datetime

# Camera initialization
frameWidth = 480
frameHeight = 480

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected with the result code " + str(rc))
        client.subscribe("CS3237/Group_22/data/images")
    else:
        print("Connection failed with error code: %d." % rc)

def on_message(client, userdata, msg):
    pass

def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    print("Connecting")
    client.username_pw_set("test", "testing")
    client.connect(hostname, 1883)
    client.loop_start()
    return client

client = setup("cs3237-v6b8lpphitxf.cedalo.cloud")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(10, 150)
#serial = serial.Serial("/dev/cu.usbserial-210", 9600, timeout=1)

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
        #data = serial.readline()
        #data = data.decode("utf-8", errors='ignore')
        #print(data)

        imgShape = len(img)
        img = img[int(imgShape*0.2):int(imgShape*0.8), int(imgShape*0.2):int(imgShape*0.8)]
        img = cv2.resize(img, (256, 256), interpolation= cv2.INTER_CUBIC)

        # Send image up through MQTT broker
        img_list = img.tolist()
        send_dict = {"filename":filename, "data":img_list, "hr":"data"}
        if type(img_list) == list:
            print("Sent data")
            client.publish("CS3237/Group_22/data/images", json.dumps(send_dict))
        if cv2.waitKey(10) & 0xFF == ord('q'):
           break
    sleep(1)