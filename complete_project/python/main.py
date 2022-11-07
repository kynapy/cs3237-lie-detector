'''
Main code used to operate. 
Subscribes to CS3237/Group_22/data/images for facial images
Subscribes to CS3237/Group_22/data/heartrate for heart rate
Subscribes to CS3237/Group_22/start for instructions
Publishes to CS3237/Group_22/start the results of the ML model

'''

import paho.mqtt.client as mqtt

collectData = False

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected with result code " +str(rc))
        client.subscribe("CS3237/Group_22/start")
    else:
        print("Connection failed with error code: %d." % rc)

def on_message(client, userdata, msg):
    print("Message received.")
    message = str(msg.payload.decode("utf-8"))
    print(message)
    if message == "start":
        # Start collecting data 
        client.subscribe("CS3237/Group_22/data/images")
        client.subscribe("CS3237/Group_22/data/heartrate")

        # Store data in temporary folder

    elif message == "stop":
        # Terminate data collection
        client.unsubscribe("CS3237/Group_22/data/images")
        client.unsubscribe("CS3237/Group_22/data/heartrate")

        # Calculate result using model
        lie = False # (TODO)

        # Return result
        if lie:
            client.publish("CS3237/Group_22/start", "lies")

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
    while True:
        pass

if __name__ == "__main__":
    main()