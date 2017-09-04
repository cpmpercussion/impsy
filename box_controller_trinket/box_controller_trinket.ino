#include <Adafruit_SoftServo.h>  // SoftwareServo (works on non PWM pins)

boolean debug = false;

Adafruit_SoftServo servoObject;

#define SERVOPIN 0
#define POTPIN 1

#define POTMIN = 560;
#define POTMAX = 1020;
#define SERVOMIN = 30;
#define SERVOMAX = 150;

byte lastPotValue;
byte lastWrittenPotValue;

int goalPosition = 127;
int servoCommand;
byte receivedPositionCommand = 0;

void setup()
{
  // Set up the interrupt that will refresh the servo for us automagically
  OCR0A = 0xAF;            // any number is OK
  TIMSK |= _BV(OCIE0A);    // Turn on the compare interrupt (below!)
  Serial.begin(9600);
  servoObject.attach(SERVOPIN);
}

void loop()
{
  byte goalPosition;
  int rawPotValue = constrain(analogRead(POTPIN), POTMIN, POTMAX);
  byte potValue = map(rawPotValue, POTMIN, POTMAX, 0, 255);
  int clonePosition = map(potValue, 0, 255, SERVOMIN, SERVOMAX);

  // DEBUG:
  if (debug == true) {
    goalPosition = potValue;
    commandServo(goalPosition);

    Serial.print("P: ");
    Serial.print(rawPotValue);
    Serial.print("(");
    Serial.print(byte(potValue), DEC);
    Serial.print("), M: ");
    Serial.println(clonePosition);
    delay(50);

    // NORMAL:
  } else {
    while (Serial.available() > 0) {
      goalPosition = Serial.read();
      commandServo(goalPosition);
    }
    // only write if changed.
    // TODO: check for out by one as well.
    if (abs(potValue - lastWrittenPotValue) > 0x02) {
      Serial.write(potValue);
      lastWrittenPotValue = potValue;
    }
    lastPotValue = potValue;
  }



}

/// Move the servo in response to input bytes
void commandServo(byte goal)
{
  servoCommand = constrain(map(goal, 0, 255, SERVOMIN, SERVOMAX), SERVOMIN, SERVOMAX);
  servoObject.write(servoCommand);
}

// We'll take advantage of the built in millis() timer that goes off
// to keep track of time, and refresh the servo every 20 milliseconds
volatile uint8_t counter = 0;
SIGNAL(TIMER0_COMPA_vect) {
  // this gets called every 2 milliseconds
  counter += 2;
  // every 20 milliseconds, refresh the servos!
  if (counter >= 20) {
    counter = 0;
    servoObject.refresh();
  }
}

