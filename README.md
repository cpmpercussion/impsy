# IMPS: The Interactive Musical Predictive System

![Predictive Musical Interaction](https://github.com/cpmpercussion/imps/raw/master/images/predictive_interaction.png)

IMPS is a system for predicting musical control data in live performance. It uses a mixture density recurrent neural network (MDRNN) to observe control inputs over multiple time steps, predicting the next value of each step, and the time that expects the next value to occur. It provides an input and output interface over OSC and can work with musical interfaces with any number of real valued inputs (we've tried from 1-8). Several interactive paradigms are supported for call-response improvisation, as well as independent operation, and "filtering" of the performer's input. Whenever you use IMPS, your input data is logged to build up a training corpus and a script is provided to train new versions of your model.

## Installation

IMPS is written in Python with Keras and TensorFlow Probability, so it should work on any platform where Tensorflow can be installed. The python requirements can be installed as follows:

    pip install -r requirements.txt

The Raspberry Pi requires some care to install matching version of TensorFlow and TensorFlow Probability, so we have provided a special requirements file:

    pip install -r pi_requirements.txt

## How to use

There are four steps for using IMPS. First, you'll need to setup your musical interface to send it OSC data and receive predictions the same way. Then you can log data, train the MDRNN, and make predictions using our provided scripts.

### 1. Connect music interface and synthesis software

### 2. Log some training data

### 3. Train an MDRNN

### 4. Perform with your predictive model

## More about Mixture Density Recurrent Neural Networks

![A Musical MDRNN](https://github.com/cpmpercussion/imps/raw/master/images/mdn_diagram.png)

