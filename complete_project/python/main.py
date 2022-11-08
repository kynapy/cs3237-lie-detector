'''
Main code used to operate. 
Subscribes to CS3237/Group_22/data/images for facial images
Subscribes to CS3237/Group_22/data/heartrate for heart rate
Subscribes to CS3237/Group_22/start for instructions
Publishes to CS3237/Group_22/start the results of the ML model

'''

import paho.mqtt.client as mqtt
import os
from datetime import datetime
import json
import numpy as np
import cv2
from test_model import Prediction

collectData = False
prediction = Prediction()
path = "data/"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected with result code " +str(rc))
        client.subscribe("CS3237/Group_22/start")
        print("Subscribed to /start", end = "\n\n")
    else:
        print("Connection failed with error code: %d." % rc)

def on_message(client, userdata, msg):
    message = str(msg.payload.decode("utf-8"))
    if message == "start":
        # Start collecting data 
        print("Start data collection...")
        client.subscribe("CS3237/Group_22/data/images")
        print("Subscribed to /data/images")
        client.subscribe("CS3237/Group_22/data/heartrate")
        print("Subscribed to /data/heartrate", end = "\n\n")
        
        # Store data in temporary folder
        global hrFile
        hrFile = open("hrData.txt", "w")      # .txt to store hr data

    elif message == "stop":
        print("Reading stopped.")

        # Terminate data collection
        client.unsubscribe("CS3237/Group_22/data/images")
        print("Unsubscribed from /data/images")
        client.unsubscribe("CS3237/Group_22/data/heartrate")
        print("Unsubscribed from /data/heartrate", end = "\n\n")
            
        # Calculate result using model
        global prediction
        result = prediction("./data")
        lie = (result<0.5)

        # Clear the data folder
        for filename in os.listdir(os.getcwd()):
            os.remove(os.path.join(os.getcwd(), filename))

        # Return result
        if lie:
            client.publish("CS3237/Group_22/start", "lie")

    elif message[0] == "{":    # Image data
        print("Image received")
        recv_dict = json.loads(msg.payload)
        img_data = np.array(recv_dict["data"])
        if img_data.any():
            cv2.imwrite(recv_dict["filename"], img_data)

    else:   # HR data
        print("Heart rate received: ")
        currentTime = 0
        count = 0
        now = datetime.now()
        if currentTime == now.strftime("%H:%M:%S"):
            count += 1
        else:
            currentTime = now.strftime("%H:%M:%S")
            count = 0
        hrFile.write("(" + currentTime + "_" + str(count) + ") : ")
        hrFile.write(message + "\n")
        hrFile.flush()
        print(message)


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    print("Connecting")
    client.connect(hostname, 1883)
    client.loop_start()
    return client

def main():
    setup("broker.emqx.io")
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory created")
    os.chdir(path)
    while True:
        pass

if __name__ == "__main__":
    main()