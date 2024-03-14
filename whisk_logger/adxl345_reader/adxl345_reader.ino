#include <Wire.h>
#include <ADXL345.h>

ADXL345 adxl; // Create an instance of the ADXL345 library

void setup() {
  Serial.begin(9600);
  Wire.begin(); // Initialize I2C communication
  adxl.powerOn(); // Power on the ADXL345
  
  // Set measurement range
  // +/-2G, +/-4G, +/-8G, +/-16G
  adxl.setRangeSetting(16);
}

void loop() {
  int x, y, z;
  adxl.readAccel(&x, &y, &z); // Read the accelerometer values
  Serial.print("X: ");
  Serial.print(x);
  Serial.print(" Y: ");
  Serial.print(y);
  Serial.print(" Z: ");
  Serial.println(z);

  delay(100); // Delay to make the output readable
}
