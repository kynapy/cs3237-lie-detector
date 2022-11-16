'''
This code is used for the camera to upload the images to CS3237/Group_22/data/images channel

'''

import cv2
import paho.mqtt.client as mqtt
from time import sleep
import json
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
    client.connect(hostname, 1883)
    client.loop_start()
    return client

def main():
    client = setup("broker.emqx.io")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap.set(10, 150)

    while cap.isOpened():
        now = datetime.now()
        currentTime = 0
        if currentTime == now.strftime("%H:%M:%S"):
            count += 1
        else:
            currentTime = now.strftime("%H:%M:%S")
            count = 0
        filename = "image_" + currentTime + "_" + str(count) +".png"
        success, img = cap.read()
        if success:
            imgShape = len(img)
            img = img[int(imgShape*0.2):int(imgShape*0.8), int(imgShape*0.2):int(imgShape*0.8)]
            img = cv2.resize(img, (256, 256), interpolation= cv2.INTER_CUBIC)

            # Send image up through MQTT broker
            img_list = img.tolist()
            send_dict = {"filename":filename, "data" : img_list}
            client.publish("CS3237/Group_22/data/images", json.dumps(send_dict))
            if cv2.waitKey(10) & 0xFF == ord('q'):
               break
        sleep(0.2)
main()