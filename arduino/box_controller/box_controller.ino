#include <Servo.h>

#define DEBUG_MODE false  // Debug mode links pot input to servo, should be false for normal use.

Servo servoObject;

#define SERVOPIN 10
#define POTPIN 0
#define POTMIN 170
#define POTMAX 852
#define SERVOMIN 5
#define SERVOMAX 175

byte lastPotValue;
byte lastWrittenPotValue;
int goalPosition = 127;
int lastGoalPosition;
int servoCommand;
int rawPotValue;
byte potValue;

byte receivedPositionCommand = 0;

void setup()
{
  Serial.begin(115200);
  servoObject.attach(SERVOPIN);
}

void loop()
{
  rawPotValue = constrain(analogRead(POTPIN), POTMIN, POTMAX);
  potValue = map(rawPotValue, POTMIN, POTMAX, 0, 255);

  // DEBUG:
  if (DEBUG_MODE) {
    int clonePosition = map(potValue, 0, 255, SERVOMIN, SERVOMAX);
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
    // Only write if values have changed.
    // Handle Servo
    while (Serial.available() > 0) {
      goalPosition = Serial.read(); 
    }
    if (abs(goalPosition - lastGoalPosition) > 2) {
      commandServo(goalPosition);
      lastGoalPosition = goalPosition;
    }
    // Handle Potentiometer
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

