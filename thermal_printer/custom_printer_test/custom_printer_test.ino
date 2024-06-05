#include "Adafruit_Thermal.h"
#include "caldron_logo.h"

// Here's the new syntax when using SoftwareSerial (e.g. Arduino Uno) ----
// If using hardware serial instead, comment out or remove these lines:

#include "SoftwareSerial.h"
#define TX_PIN 1 // Arduino transmit  YELLOW WIRE  labeled RX on printer
#define RX_PIN 0 // Arduino receive   GREEN WIRE   labeled TX on printer

SoftwareSerial mySerial(RX_PIN, TX_PIN); // Declare SoftwareSerial obj first
//Adafruit_Thermal printer(&mySerial);     // Pass addr to printer constructor
// Then see setup() function regarding serial & printer begin() calls.

// Here's the syntax for hardware serial (e.g. Arduino Due) --------------
// Un-comment the following line if using hardware serial:

Adafruit_Thermal printer(&Serial1);      // Or Serial2, Serial3, etc.

// -----------------------------------------------------------------------

void setup() {

  pinMode(7, OUTPUT); digitalWrite(7, LOW);

  Serial1.begin(9600); // Use this instead if using hardware serial
  printer.begin();        // Init printer (same regardless of serial type)

  char welcome_msg[] = "Hi! Thanks for using Caldron.\nYour recipe is printed below.";
  char recipe[] = "Chocolate Chip Cookies";
  char ingredients_str[] = "butter, 1.0 cup,\ngranulated sugar, 1.0 cup,\nlight brown sugar, 1.0 cup,\npure vanilla extract, 2.0 teaspoons,\neggs, 2.0 large,\nall-purpose flour, 3.0 cups,\nbaking soda, 1.0 teaspoon,\nbaking powder, 0.5 teaspoon,\nsea salt, 1.0 teaspoon,\nchocolate chips, 2.0 cups";
  char instructions_str[] = "Preheat oven to 375 degrees F.\nCream together butter and sugars, then add eggs and vanilla.\nMix in dry ingredients, then add chocolate chips.\nRoll dough into balls and bake for 8-10 minutes.\nLet cool before serving.";

  printer.setFont('A');
  printer.boldOn();
  printer.justify('L');
  printer.printBitmap(caldron_logo_width, caldron_logo_height, caldron_logo_data);

  printer.sleep();      // Tell printer to sleep
  delay(1000L);         // Sleep for 3 seconds
  printer.wake();       // MUST wake() before printing again, even if reset

  printer.underlineOn();
  printer.doubleHeightOn();
  printer.print(F("Recipe:"));
  printer.underlineOff();
  printer.println(recipe);
  printer.doubleHeightOff();
  printer.println(F(""));

  printer.underlineOn();
  printer.doubleHeightOn();
  printer.print(F("Ingredients:"));
  printer.underlineOff();
  printer.doubleHeightOff();
  printer.println(F(""));

  printer.println(ingredients_str);
  printer.println(F(""));

  printer.underlineOn();
  printer.doubleHeightOn();
  printer.print(F("Instructions:"));
  printer.underlineOff();
  printer.doubleHeightOff();
  printer.println(F(""));

  printer.println(instructions_str);
  printer.println(F(""));

  printer.printBitmap(caldron_logo_width, caldron_logo_height, caldron_logo_data);
  printer.justify('C');
  printer.println(F("Caldron"));

  // Print the 135x135 pixel QR code in adaqrcode.h:
  printer.feed(2);

  printer.sleep();      // Tell printer to sleep
  delay(3000L);         // Sleep for 3 seconds
  printer.wake();       // MUST wake() before printing again, even if reset
  printer.setDefault(); // Restore printer to defaults
}

void loop() {
}
