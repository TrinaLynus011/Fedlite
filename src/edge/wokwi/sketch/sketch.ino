#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ============================================
// FEDLITE-CARE HOSPITAL EDGE NODE
// ESP32 Wokwi Simulation
// ============================================

// ---------- OLED ----------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

#define OLED_SDA 21
#define OLED_SCL 22

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  -1
);

// ---------- HARDWARE PINS ----------
const int STATUS_LED = 2;
const int STATE_BUTTON = 4;

// ---------- NODE STATES ----------
enum NodeState {
  NODE_AVAILABLE,
  NODE_BUSY,
  NODE_CONSTRAINED,
  NODE_OFFLINE
};

NodeState currentState = NODE_AVAILABLE;

// ---------- TELEMETRY ----------
int cpuLoad = 42;
int memoryPressure = 31;
int networkQuality = 82;

// ---------- BUTTON ----------
bool lastButtonState = HIGH;

// ---------- TIMING ----------
unsigned long lastTelemetryTime = 0;
unsigned long lastBlinkTime = 0;

const unsigned long TELEMETRY_INTERVAL = 3000;


// ============================================
// SETUP
// ============================================

void setup() {

  Serial.begin(115200);

  pinMode(STATUS_LED, OUTPUT);
  pinMode(STATE_BUTTON, INPUT_PULLUP);

  Wire.begin(OLED_SDA, OLED_SCL);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED initialization failed!");
    while (true);
  }

  Serial.println();
  Serial.println("========================================");
  Serial.println("   FEDLITE-CARE EDGE NODE SIMULATION");
  Serial.println("========================================");
  Serial.println("Node ID: Hospital-Edge-01");
  Serial.println("Status: INITIALIZING...");

  showStartupScreen();

  delay(2000);

  Serial.println("Status: ONLINE");

  printTelemetry();
  updateDisplay();
}


// ============================================
// MAIN LOOP
// ============================================

void loop() {

  handleButton();
  updateLED();

  unsigned long currentTime = millis();

  if (currentTime - lastTelemetryTime >= TELEMETRY_INTERVAL) {

    simulateTelemetry();
    printTelemetry();
    updateDisplay();

    lastTelemetryTime = currentTime;
  }
}


// ============================================
// BUTTON HANDLING
// ============================================

void handleButton() {

  bool buttonState = digitalRead(STATE_BUTTON);

  if (lastButtonState == HIGH && buttonState == LOW) {

    switch (currentState) {

      case NODE_AVAILABLE:
        currentState = NODE_BUSY;
        break;

      case NODE_BUSY:
        currentState = NODE_CONSTRAINED;
        break;

      case NODE_CONSTRAINED:
        currentState = NODE_OFFLINE;
        break;

      case NODE_OFFLINE:
        currentState = NODE_AVAILABLE;
        break;
    }

    Serial.println();
    Serial.println(">>> STATE CHANGED <<<");
    Serial.print("New State: ");
    Serial.println(getStateName());

    updateDisplay();

    delay(200);
  }

  lastButtonState = buttonState;
}


// ============================================
// LED CONTROL
// ============================================

void updateLED() {

  unsigned long currentTime = millis();

  switch (currentState) {

    case NODE_AVAILABLE:

      digitalWrite(STATUS_LED, HIGH);
      break;


    case NODE_BUSY:

      if (currentTime - lastBlinkTime >= 800) {

        digitalWrite(
          STATUS_LED,
          !digitalRead(STATUS_LED)
        );

        lastBlinkTime = currentTime;
      }

      break;


    case NODE_CONSTRAINED:

      if (currentTime - lastBlinkTime >= 200) {

        digitalWrite(
          STATUS_LED,
          !digitalRead(STATUS_LED)
        );

        lastBlinkTime = currentTime;
      }

      break;


    case NODE_OFFLINE:

      digitalWrite(STATUS_LED, LOW);
      break;
  }
}


// ============================================
// SIMULATE TELEMETRY
// ============================================

void simulateTelemetry() {

  switch (currentState) {

    case NODE_AVAILABLE:

      cpuLoad = random(25, 50);
      memoryPressure = random(20, 45);
      networkQuality = random(70, 100);

      break;


    case NODE_BUSY:

      cpuLoad = random(60, 80);
      memoryPressure = random(50, 70);
      networkQuality = random(55, 80);

      break;


    case NODE_CONSTRAINED:

      cpuLoad = random(80, 98);
      memoryPressure = random(75, 95);
      networkQuality = random(20, 55);

      break;


    case NODE_OFFLINE:

      cpuLoad = 0;
      memoryPressure = 0;
      networkQuality = 0;

      break;
  }
}


// ============================================
// SERIAL TELEMETRY
// ============================================

void printTelemetry() {

  Serial.println();

  Serial.println("----- EDGE NODE TELEMETRY -----");
  Serial.println("Node ID: Hospital-Edge-01");

  Serial.print("Node State: ");
  Serial.println(getStateName());

  Serial.print("CPU Load: ");
  Serial.print(cpuLoad);
  Serial.println("%");

  Serial.print("Memory Pressure: ");
  Serial.print(memoryPressure);
  Serial.println("%");

  Serial.print("Network Quality: ");
  Serial.print(networkQuality);
  Serial.println("%");

  Serial.print("Participation Status: ");
  Serial.println(getParticipationStatus());

  // Machine-readable telemetry record for the local JSONL ingest path.
  // This is simulated device telemetry only; it does not represent a live
  // physical ESP32 network connection or host-machine measurement.
  Serial.print("JSON: ");
  Serial.print("{\"telemetry_version\":1,\"node_id\":\"Hospital-Edge-01\",\"state\":\"");
  Serial.print(getStateName());
  Serial.print("\",\"cpu_load\":");
  Serial.print(cpuLoad);
  Serial.print(",\"memory_pressure\":");
  Serial.print(memoryPressure);
  Serial.print(",\"network_quality\":");
  Serial.print(networkQuality);
  Serial.print(",\"participation_status\":\"");
  Serial.print(getParticipationStatus());
  Serial.print("\",\"sequence_id\":");
  Serial.print((millis() / 3000) + 1);
  Serial.print(",\"timestamp_ms\":");
  Serial.print(millis());
  Serial.println(",\"source\":\"mcu_simulation\"}");

  Serial.println("-------------------------------");
}


// ============================================
// OLED DISPLAY
// ============================================

void updateDisplay() {

  display.clearDisplay();

  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  display.setCursor(0, 0);
  display.println("FEDLITE-CARE");
  display.println("EDGE NODE 01");
  display.println("---------------------");

  display.print("State: ");
  display.println(getStateName());

  display.print("CPU: ");
  display.print(cpuLoad);
  display.println("%");

  display.print("MEM: ");
  display.print(memoryPressure);
  display.println("%");

  display.print("NET: ");
  display.print(networkQuality);
  display.println("%");

  display.print("FL: ");
  display.println(getParticipationStatus());

  display.display();
}


// ============================================
// STARTUP SCREEN
// ============================================

void showStartupScreen() {

  display.clearDisplay();

  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  display.setCursor(10, 15);
  display.println("FEDLITE-CARE");

  display.setCursor(10, 30);
  display.println("INITIALIZING...");

  display.display();
}


// ============================================
// STATE NAME
// ============================================

String getStateName() {

  switch (currentState) {

    case NODE_AVAILABLE:
      return "AVAILABLE";

    case NODE_BUSY:
      return "BUSY";

    case NODE_CONSTRAINED:
      return "CONSTRAINED";

    case NODE_OFFLINE:
      return "OFFLINE";

    default:
      return "UNKNOWN";
  }
}


// ============================================
// PARTICIPATION STATUS
// ============================================

String getParticipationStatus() {

  switch (currentState) {

    case NODE_AVAILABLE:
      return "READY";

    case NODE_BUSY:
      return "LIMITED";

    case NODE_CONSTRAINED:
      return "RESTRICTED";

    case NODE_OFFLINE:
      return "OFFLINE";

    default:
      return "UNKNOWN";
  }
}