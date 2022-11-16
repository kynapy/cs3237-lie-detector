#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <PubSubClient.h>

// HR measurement
#define samp_siz 4
#define rise_threshold 5
#define sensorPin A0

// WiFi setup
const char* ssid = "kynapy";
const char* password = "KYNAPY123";

const char* mqtt_broker = "broker.emqx.io";
const char* topic = "CS3237/Group_22/data/heartrate";
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
  Serial.begin(9600);
  connectToWifi();
  connectToMqtt();
  //client.publish(topic, "Joined Group_22/data");
  //client.subscribe(topic);
}

void callback(char* topic, byte* payload, unsigned int length) {
  int fill = 0;
}

void loop() {
  // put your main code here, to run repeatedly:
  client.loop();
  float reads[samp_siz], sum;
  long int now, ptr;
  float last, reader, start;
  float first, second, third, before, print_value;
  bool rising;
  int rise_count;
  int n;
  long int last_beat;
  for (int i = 0; i < samp_siz; i++)
    reads[i] = 0;
  sum = 0;
  ptr = 0;
  while(1)
  {
    n = 0;
    start = millis();
    reader = 0.;
    do
    {
      reader += analogRead(sensorPin);
      n++;
      now = millis();
    }
   while (now < start + 20);
   reader /= n;  // we got an average
    // Add the newest measurement to an array
    // and subtract the oldest measurement from the array
    // to maintain a sum of last measurements
    sum -= reads[ptr];
    sum += reader;
    reads[ptr] = reader;
    last = sum / samp_siz;
     if (last > before)
     {
      rise_count++;
      if (!rising && rise_count > rise_threshold)
      {
        delay(400);
        rising = true;
        first = millis() - last_beat;
        last_beat = millis();
        print_value = 60000. / (0.4 * first + 0.3 * second + 0.3 * third);
        char hrValue[10];
        dtostrf(print_value, 4, 4, hrValue);
        Serial.println(hrValue);
        //client.publish(topic, hrValue);
        third = second;
        second = first;
      }
    }
    else
    {
      // Ok, the curve is falling
      rising = false;
      rise_count = 0;
    }
    before = last;
    ptr++;
    ptr %= samp_siz;
  }
}
