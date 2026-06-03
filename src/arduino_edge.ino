/*
 * Thesis Project: Edge AI Anomaly Detection on IoT Environments
 * Author: Dimitriοs Kostoulas
 * Hardware: Arduino MKR WiFi 1010 + MKR IoT Carrier
 * * Description: 
 * Samples indoor/outdoor temperature every 10 minutes. 
 * Computes dynamic features (RoC, Thermal Gap) and runs a lightweight 
 * linear proxy of an Isolation Forest model to detect thermal anomalies. (Os dokimi gia teliko vhma )
 */

#include <SPI.h>
#include <WiFiNINA.h>
#include <ArduinoHttpClient.h>
#include <ArduinoJson.h>
#include <Arduino_MKRIoTCarrier.h>
#include <ThingSpeak.h>
#include <Arduino_PMIC.h>
#include <ArduinoLowPower.h>

// --- Configuration ---
const char WIFI_SSID[]      = "GK";
const char WIFI_PASS[]      = "20101971";
const char OWM_API_KEY[]    = "f9075e47178269dd099e72dfacbd51c3";
const char OWM_CITY[]       = "Athens,GR";

const unsigned long TS_CHANNEL    = 3367555UL;
const char TS_WRITE_KEY[]         = "4H9P76SPFPE26CHE";

// 10-minute sampling interval
const unsigned long SAMPLE_INTERVAL_MS  = 600000UL;
const int OWM_REFRESH_CYCLES = 6; 

// --- Isolation Forest Proxy Model Parameters ---
// Extracted from Python robust scaling & linear approximation (97.4% accuracy)
const float MEDIAN[5] = { -0.01617100f, -0.05764850f,  0.05483837f,  0.01319650f,  0.00084194f };
const float IQR[5]    = {  0.11225328f,  0.29780253f,  0.07995879f,  0.19582350f,  0.08190569f };

const float PROXY_INTERCEPT   = -0.32847773f;
const float PROXY_W[5]        = { -0.01053665f, -0.01960114f, -0.01871380f, -0.02435940f, -0.02011853f };
const float PROXY_THRESHOLD   = -0.52535224f;

// --- Circular Buffer ---
const int RING_SIZE    = 5;  
const int WARMUP_COUNT = 4;  

float indoorBuf[RING_SIZE];  
float gapBuf[RING_SIZE];     
int   bufHead    = 0;        
int   readingN   = 0;        

// --- Global State ---
MKRIoTCarrier carrier;
WiFiSSLClient  wifiSSL;
WiFiClient     clientTS;
HttpClient     owmClient = HttpClient(wifiSSL, "api.openweathermap.org", 443);

float  externalTemp      = 20.0f;  
float  indoorHumidity    = 0.0f;
int    cyclesSinceOWM    = OWM_REFRESH_CYCLES; 
int    batteryPct        = 100;
bool   lastWasAnomaly    = false;
float  lastAnomalyScore  = 0.0f;

// --- Helper Functions ---
inline int prevIdx(int head, int stepsBack) {
  return (head - stepsBack + RING_SIZE) % RING_SIZE;
}

float robustScale(float x, int featureIdx) {
  return (x - MEDIAN[featureIdx]) / IQR[featureIdx];
}

float computeAnomalyScore(float indoorNow, float indoorPrev1,
                          float indoorPrev3, float indoorStd3,
                          float gapNow,     float gapPrev1) {
  
  float roc1 = indoorNow - indoorPrev1;
  float roc3 = indoorNow - indoorPrev3;
  float rstd = indoorStd3;
  float dgap = gapNow - gapPrev1;
  
  float roc1_prev = indoorPrev1 - indoorPrev3;   
  float accel     = roc1 - (roc1_prev / 2.0f);   

  float features[5] = { roc1, roc3, rstd, dgap, accel };

  float score = PROXY_INTERCEPT;
  for (int i = 0; i < 5; i++) {
    float z = robustScale(features[i], i);
    score += PROXY_W[i] * fabsf(z);
  }
  return score;
}

void readBattery() {
  int raw = analogRead(ADC_BATTERY);
  batteryPct = constrain(map(raw, 750, 1023, 0, 100), 0, 100);
}

bool wifiConnect() {
  if (WiFi.status() == WL_CONNECTED) return true;
  int attempts = 0;
  while (WiFi.begin(WIFI_SSID, WIFI_PASS) != WL_CONNECTED && attempts < 5) {
    delay(2000);
    attempts++;
  }
  return (WiFi.status() == WL_CONNECTED);
}

void wifiDisconnect() {
  WiFi.disconnect();
  WiFi.end();
}

void fetchExternalWeather() {
  if (!wifiConnect()) return;

  String path = "/data/2.5/weather?q=" + String(OWM_CITY) +
                "&appid=" + String(OWM_API_KEY) + "&units=metric";
  owmClient.get(path);

  if (owmClient.responseStatusCode() == 200) {
    String body = owmClient.responseBody();
    DynamicJsonDocument doc(1536);
    if (!deserializeJson(doc, body)) {
      externalTemp = doc["main"]["temp"];
    }
  }
  owmClient.stop();
}

void sendAnomalyToThingSpeak(float indoor, float outdoor,
                              float score,  float humid, int bat) {
  if (!wifiConnect()) return;

  ThingSpeak.begin(clientTS);
  ThingSpeak.setField(1, indoor);    
  ThingSpeak.setField(2, outdoor);   
  ThingSpeak.setField(3, score);     
  ThingSpeak.setField(4, humid);     
  ThingSpeak.setField(5, bat);       

  int result = ThingSpeak.writeFields(TS_CHANNEL, TS_WRITE_KEY);

  if (result == 200) {
    carrier.display.fillScreen(ST77XX_GREEN);
  } else {
    carrier.display.fillScreen(ST77XX_RED);
  }
  delay(800);
  carrier.display.fillScreen(ST77XX_BLACK);
}

void showDisplay(float indoor, float outdoor, float humid, bool anomaly, float score) {
  carrier.display.fillScreen(ST77XX_BLACK);
  carrier.display.setTextSize(2);

  if (anomaly) {
    carrier.display.fillRect(0, 0, 240, 36, ST77XX_RED);
    carrier.display.setTextColor(ST77XX_WHITE);
    carrier.display.setCursor(30, 10);
    carrier.display.print("!! ANOMALY !!");
    carrier.display.setTextColor(ST77XX_RED);
  } else {
    carrier.display.setTextColor(ST77XX_GREEN);
    carrier.display.setCursor(50, 10);
    carrier.display.print("NORMAL");
    carrier.display.setTextColor(ST77XX_WHITE);
  }

  carrier.display.setTextColor(ST77XX_CYAN);
  carrier.display.setCursor(10, 55);
  carrier.display.print("IN:  ");
  carrier.display.print(indoor, 1);
  carrier.display.print(" C");

  carrier.display.setTextColor(ST77XX_ORANGE);
  carrier.display.setCursor(10, 85);
  carrier.display.print("OUT: ");
  carrier.display.print(outdoor, 1);
  carrier.display.print(" C");

  carrier.display.setTextColor(ST77XX_YELLOW);
  carrier.display.setCursor(10, 115);
  carrier.display.print("GAP: ");
  carrier.display.print(indoor - outdoor, 1);
  carrier.display.print(" C");

  carrier.display.setTextColor(0x867D);  
  carrier.display.setCursor(10, 145);
  carrier.display.print("HUM: ");
  carrier.display.print(humid, 1);
  carrier.display.print(" %");

  carrier.display.setTextColor(ST77XX_WHITE);
  carrier.display.setCursor(10, 170);
  carrier.display.print("SCR: ");
  carrier.display.print(score, 3);

  carrier.display.setTextColor(batteryPct > 20 ? ST77XX_GREEN : ST77XX_RED);
  carrier.display.setCursor(10, 195);
  carrier.display.print("BAT: ");
  carrier.display.print(batteryPct);
  carrier.display.print("%");

  carrier.display.setTextColor(0x4A69);  
  carrier.display.setTextSize(1);
  carrier.display.setCursor(10, 210);
  carrier.display.print("Reading #");
  carrier.display.print(readingN);

  delay(4000);   
  carrier.display.fillScreen(ST77XX_BLACK);
}

void setup() {
  Serial.begin(9600);

  if (PMIC.begin()) {
    PMIC.setInputCurrentLimit(2.0f);
  }

  CARRIER_CASE = true;
  carrier.begin();
  carrier.display.fillScreen(ST77XX_BLACK);
  carrier.display.setTextColor(ST77XX_WHITE);
  carrier.display.setTextSize(2);
  carrier.display.setCursor(20, 100);
  carrier.display.print("Initializing...");

  for (int i = 0; i < RING_SIZE; i++) {
    indoorBuf[i] = 0.0f;
    gapBuf[i]    = 0.0f;
  }

  delay(2000);
  carrier.display.fillScreen(ST77XX_BLACK);
}

void loop() {
  // 1. Read sensors
  float indoorNow    = carrier.Env.readTemperature() - 4.0f;
  indoorHumidity     = carrier.Env.readHumidity();
  readBattery();

  // 2. Fetch outdoor weather (hourly)
  cyclesSinceOWM++;
  if (cyclesSinceOWM >= OWM_REFRESH_CYCLES) {
    fetchExternalWeather();
    cyclesSinceOWM = 0;
  }

  float gapNow = indoorNow - externalTemp;

  // 3. Update ring buffer
  indoorBuf[bufHead] = indoorNow;
  gapBuf[bufHead]    = gapNow;
  bufHead = (bufHead + 1) % RING_SIZE;
  readingN++;

  // 4. Compute features & anomaly score
  bool  isAnomaly = false;
  float score     = PROXY_INTERCEPT; 

  if (readingN >= WARMUP_COUNT) {
    int iNow   = prevIdx(bufHead, 1); 
    int iPrev1 = prevIdx(bufHead, 2); 
    int iPrev2 = prevIdx(bufHead, 3); 
    int iPrev3 = prevIdx(bufHead, 4); 

    float iNowV   = indoorBuf[iNow];
    float iPrev1V = indoorBuf[iPrev1];
    float iPrev3V = indoorBuf[iPrev3];
    float gNowV   = gapBuf[iNow];
    float gPrev1V = gapBuf[iPrev1];

    float vals[3] = { iNowV, iPrev1V, indoorBuf[iPrev2] };
    float mean3   = (vals[0] + vals[1] + vals[2]) / 3.0f;
    float var3    = ((vals[0]-mean3)*(vals[0]-mean3) +
                     (vals[1]-mean3)*(vals[1]-mean3) +
                     (vals[2]-mean3)*(vals[2]-mean3)) / 3.0f;
    float std3    = sqrtf(var3);

    score = computeAnomalyScore(iNowV, iPrev1V, iPrev3V, std3, gNowV, gPrev1V);
    isAnomaly = (score < PROXY_THRESHOLD);
  }

  // 5. Update local display
  showDisplay(indoorNow, externalTemp, indoorHumidity, isAnomaly, score);

  // 6. Push to ThingSpeak only if anomaly detected
  if (isAnomaly) {
    sendAnomalyToThingSpeak(indoorNow, externalTemp, score, indoorHumidity, batteryPct);
  }

  // 7. Ensure WiFi is off to save battery
  if (WiFi.status() == WL_CONNECTED) {
    wifiDisconnect();
  }

  // --- Debug Logging ---
  Serial.print("Reading: "); Serial.print(readingN);
  Serial.print(" | Temp: "); Serial.print(indoorNow);
  Serial.print(" | Score: "); Serial.print(score, 4);
  Serial.println(isAnomaly ? " [ANOMALY DETECTED]" : " [NORMAL]");

  // 8. Deep Sleep Configuration
  analogWrite(TFT_BACKLIGHT, 0); 
  LowPower.deepSleep(SAMPLE_INTERVAL_MS);
  analogWrite(TFT_BACKLIGHT, 255); 
}
