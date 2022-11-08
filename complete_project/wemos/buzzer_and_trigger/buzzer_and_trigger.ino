#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <PubSubClient.h>

// Button Sensor
#define BUTTON D5
#define speakerPin D6

// WiFi
const char* ssid = "kynapy";
const char* password = "KYNAPY123";

const char* mqtt_broker = "broker.emqx.io";
const char* topic = "CS3237/Group_22/start";
const int mqtt_port = 1883;

volatile byte state = LOW;
volatile static unsigned long last_isr_time = 0;

WiFiClient espClient;
PubSubClient client(espClient);

IRAM_ATTR void toggle() {
  unsigned long interrupt_time = millis();
  if (interrupt_time - last_isr_time > 200) {
    if (!state){
      client.publish(topic, "start");
    } else {
      client.publish(topic, "stop");
    }
    state = !state;
    Serial.println("Button pressed");
    last_isr_time = interrupt_time;
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  buzz();
}

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
  Serial.begin(9600);
  connectToWifi();
  connectToMqtt();
  pinMode(speakerPin, OUTPUT);
  pinMode(BUTTON, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON),toggle, RISING);
  //client.publish(topic, "Joined Group_22/data");
  client.subscribe("CS3237/Group_22/result");
}

void buzz() {
  analogWrite(speakerPin,255);
  //buzzTime = millis();
  delay(2000);
  analogWrite(speakerPin, 0);
}

void loop() {
  // put your main code here, to run repeatedly:
  client.loop();
}
