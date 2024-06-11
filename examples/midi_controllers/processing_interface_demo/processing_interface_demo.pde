import oscP5.*;
import netP5.*;
import ddf.minim.*;
import ddf.minim.ugens.*;

float xloc;
float yloc;
boolean displayed;
  
OscP5 oscP5;
NetAddress impsAddress;

Minim minim;
AudioOutput out;
Oscil       wave;
 
void setup() {
  size(640, 480); 
  noStroke();
  rectMode(CENTER);
  oscP5 = new OscP5(this,5000);
  impsAddress = new NetAddress("localhost",5001);
  minim = new Minim(this);
  out = minim.getLineOut();
  wave = new Oscil( 440, 0.0, Waves.SINE );
  wave.patch( out );
}

void draw() {
  background(51); 
  fill(255, 204);
  // visualise mouse
  fill(204, 102, 0);
  rect(mouseX, height/2, mouseY/2+10, mouseY/2+10);
  // visualise imps
  fill(102, 204, 0);
  rect(xloc*width, height/2, (yloc*height/2)+10, (yloc*height/2)+10);
}

void mouseMoved() {
  OscMessage myMessage = new OscMessage("/interface");
  float x = float(mouseX) / float(width);
  float y = float(mouseY) / float(height);
  myMessage.add(x); /* add an int to the osc message */
  myMessage.add(y);
  oscP5.send(myMessage, impsAddress); // send to imps
  // sonify
  float amp = map( y, 1.0, 0, 1, 0 );
  wave.setAmplitude( amp );
  float freq = map( x, 0, 1.0, 110, 880 );
  wave.setFrequency( freq );
}

/* play back and visualise incoming data */
void oscEvent(OscMessage theOscMessage) {
  xloc = theOscMessage.get(0).floatValue();
  yloc = theOscMessage.get(1).floatValue();
  float amp = map( yloc, 1.0, 0, 1, 0 );
  wave.setAmplitude( amp );
  float freq = map( xloc, 0, 1.0, 110, 880 );
  wave.setFrequency( freq );
}
