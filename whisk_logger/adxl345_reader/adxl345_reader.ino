#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

void setup() {
  Serial.begin(9600);
  while (!Serial); // Wait for serial port to connect.

  if (!accel.begin()) {
    Serial.println("No ADXL345 detected");
    while (1);
  }

  accel.setRange(ADXL345_RANGE_16_G);
}

void loop() {
  sensors_event_t event;
  accel.getEvent(&event);

  // Get current time in milliseconds
  unsigned long now = millis();

  // Format and print data
  Serial.print("Time:"); Serial.print(now);
  Serial.print(",X:"); Serial.print(event.acceleration.x);
  Serial.print(",Y:"); Serial.print(event.acceleration.y);
  Serial.print(",Z:"); Serial.print(event.acceleration.z);
  Serial.println("");

  delay(500); // Delay for half a second
}
