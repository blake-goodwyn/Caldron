#include <Wire.h>
#include "Adafruit_MPR121.h"

#ifndef _BV
#define _BV(bit) (1 << (bit)) 
#endif

// You can have up to 4 on one i2c bus but one is enough for testing!
Adafruit_MPR121 cap = Adafruit_MPR121();

// Keeps track of the last pins touched
// so we know when buttons are 'released'
uint16_t lasttouched = 0;
uint16_t currtouched = 0;

void setup() {
  Serial.begin(115200);

  while (!Serial) { // needed to keep leonardo/micro from starting too fast!
    delay(10);
  }
  
  Serial.println("Adafruit MPR121 Capacitive Touch sensor test"); 
  
  // Default address is 0x5A, if tied to 3.3V its 0x5B
  // If tied to SDA its 0x5C and if SCL then 0x5D
  if (!cap.begin(0x5B)) {
    Serial.println("MPR121 not found, check wiring?");
    while (1);
  }
  Serial.println("MPR121 found!");
}

void loop() {

  // Append each channel's data to the line
  for (uint8_t i = 0; i < 12; i++) {
    Serial.print(i); Serial.print("\t");
    Serial.print(cap.filteredData(i)); Serial.print("\t");
    Serial.print(cap.baselineData(i));
    if (i < 11) {
      Serial.print("\t"); // Tab separator for all but last channel
    }
  }

  // End the line
  Serial.println();

  delay(100); // Adjust or remove delay as needed
}


