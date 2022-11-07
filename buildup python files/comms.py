import paho.mqtt.client as mqtt
from time import sleep

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected with result code " +str(rc))
        client.subscribe("CS3237/Group_22/data")
    else:
        print("Connection failed with error code: %d." % rc)

def on_message(client, userdata, msg):
    print("Received message: ", end="")
    message = str(msg.payload.decode("utf-8"))
    print(message)
    #result = toFahrenheit(float(message))
    #print("Sending results", end = "\n\n")
    #client.publish("CS3237/Group_22/classification", result)

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