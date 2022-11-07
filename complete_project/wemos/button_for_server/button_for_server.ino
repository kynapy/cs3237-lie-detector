#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <PubSubClient.h>

// Button Sensor
#define BUTTON D5

// WiFi
const char* ssid2 = "kynapy";
const char* password2 = "KYNAPY123";

const char* mqtt_broker = "broker.emqx.io";
const char* topic = "CS3237/Group_22/start";
const int mqtt_port = 1883;

volatile byte state = LOW;

WiFiClient espClient;
PubSubClient client(espClient);

void connectToWifi() {
  WiFi.begin(ssid2, password2);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connecting to WiFi...");
  }
  Serial.print("Connected to: ");
  Serial.println(ssid2);
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

IRAM_ATTR void toggle(){
  static unsigned long last_interrupt_time = 0;
  unsigned long interrupt_time = millis();
  if (interrupt_time - last_interrupt_time > 200) {
    if(state){
      client.publish(topic, "Start read");
      Serial.println((String)topic + " => on");
      //start taking pictures
      state=!state;
      }
    else{
      client.publish(topic, "off");
      Serial.println((String)topic + " => off");
      state=!state;
      }
   }
   last_interrupt_time = interrupt_time;
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  connectToWifi();
  connectToMqtt();
  pinMode(BUTTON, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON),toggle, RISING);
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
