#include <Servo.h>

boolean debug = false;

Servo servoObject;

int potPin = 0;
int potMin = 560;
int potMax = 1020;
int servoMin = 30;
int servoMax = 150;

byte lastPotValue;
byte lastWrittenPotValue;

int goalPosition = 127;
int servoCommand;

byte receivedPositionCommand = 0;

void setup()
{
  Serial.begin(9600);
  servoObject.attach(12);
}

void loop()
{
  byte goalPosition;
  int rawPotValue = constrain(analogRead(potPin), potMin, potMax);
  byte potValue = map(rawPotValue, potMin, potMax, 0, 255);
  int clonePosition = map(potValue, 0, 255, servoMin, servoMax);

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
  servoCommand = constrain(map(goal, 0, 255, servoMin, servoMax), servoMin, servoMax);
  servoObject.write(servoCommand);
}

