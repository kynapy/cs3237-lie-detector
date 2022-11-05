#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <PubSubClient.h>

// WiFi setup
const char* ssid = "kynapy";
const char* password = "KYNAPY123";

const char* mqtt_broker = "broker.emqx.io";
const char* topic = "CS3237/Group_22/data";
const int mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);

void connectToWifi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connecting to WiFi...");
  }
  Serial.print("Connected to: ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void connectToMqtt() {
  Serial.println("Connecting to MQTT...");
  client.setServer(mqtt_broker, mqtt_port);
  client.setCallback(callback);
  while (!client.connected()) {
    String client_id = "esp8266-client-";
    client_id += String(WiFi.macAddress());
    Serial.printf("The client %s connects to the public mqtt broker\n", client_id.c_str());
    if (client.connect(client_id.c_str())) {
      delay(200);
    } else {
      Serial.print("failed with state ");
      Serial.println(client.state());
      delay(2000);
    } 
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  connectToWifi();
  connectToMqtt();
  //client.publish(topic, "Joined Group_22/data");
  client.subscribe(topic);
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived in topic: ");
  Serial.println(topic);
  Serial.print("Message: ");
  for (int i = 0; i < length; i++){
    Serial.print((char) payload[i]);
  }
  Serial.println();
  Serial.println("________________");
}

void loop() {
  // put your main code here, to run repeatedly:
  
}
