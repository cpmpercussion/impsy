# IMPS: The Interactive Musical Predictive System

![MIT License](https://img.shields.io/github/license/cpmpercussion/keras-mdn-layer.svg?style=flat)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2580176.svg)](https://doi.org/10.5281/zenodo.2580176)

![Predictive Musical Interaction](https://github.com/cpmpercussion/imps/raw/master/images/predictive_interaction.png)

IMPS is a system for predicting musical control data in live performance. It uses a mixture density recurrent neural network (MDRNN) to observe control inputs over multiple time steps, predicting the next value of each step, and the time that expects the next value to occur. It provides an input and output interface over OSC and can work with musical interfaces with any number of real valued inputs (we've tried from 1-8). Several interactive paradigms are supported for call-response improvisation, as well as independent operation, and "filtering" of the performer's input. Whenever you use IMPS, your input data is logged to build up a training corpus and a script is provided to train new versions of your model.

Here's a [demonstration video showing how IMPS can be used with different musical interfaces:](https://www.youtube.com/embed/Kdmhrp2dfHw)

<!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/Kdmhrp2dfHw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->

## Installation

IMPS is written in Python with Keras and TensorFlow Probability, so it should work on any platform where Tensorflow can be installed. Python 3 is required. IMPS currently relies on TensorFlow 2.0.0, TensorFlow Probability 0.8.0, and keras-mdn-layer 0.3.0.

First you should clone this repository or download it to your computer:

    git clone https://github.com/cpmpercussion/imps.git
    cd imps

The python requirements can be installed as follows:

    pip install -r requirements.txt

Some people like to keep Python packages separate in virtual environments, if that's you, here's some terminal commands to install:

    virtualenv --system-site-packages -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt

## How to use

There are four steps for using IMPS. First, you'll need to setup your musical interface to send it OSC data and receive predictions the same way. Then you can log data, train the MDRNN, and make predictions using our provided scripts.

### 1. Connect music interface and synthesis software

You'll need:

- A music interface that can output data as OSC.
- Some synthesiser software that can take OSC as input.

These could be the same piece of software or hardware!

You need to decide on the number of inputs (or dimension) for your predictive model. This is the number of continuous outputs from your interface plus one (for time). So for an interface with 8 faders, the dimension will be 9.

Now you need your music interface to send messages to IMPS over OSC. The default address for IMPS is: localhost:5001. The messages to IMPS should have the OSC address `/interface`, and then a float between 0 and 1 for each continuous output on your interface, e.g.:

    /interface 0 0.5 0.23 0.87 0.9 0.7 0.45 0.654

For an 8-dimensional interface.

Your synthesiser software or interface needs to listen for messages from the IMPS system as well. These have the same format with the OSC address `/prediction`. You can interpret these as interactions predicted to occur right when the message is sent.

Here's an example diagram for our 8-controller example, the [xtouch mini controller](https://www.musictribe.com/Categories/Behringer/Computer-Audio/Desktop-Controllers/X-TOUCH-MINI/p/P0B3M).

![Predictive Musical Interaction](https://github.com/cpmpercussion/imps/raw/master/images/IMPS_connection_example.png)

In this example we've used Pd to connect the xtouch mini to IMPS and to synthesis sounds. Our Pd mapping patch takes data from the xtouch and sends `/interface` OSC messages to IMPS, it also receives `/prediction` OSC message back from IMPS whenever they occur. Of course, whenever the user performs with the controller, the mapping patch sends commands to the synthesiser patch to make sound. Whenever `/prediction` messages are received, these also trigger changes in the synth patch, and we also send MIDI messages back to the xtouch controller to update its lights so that the performer knows what IMPS is predicting.

So what happens if IMPS and the performer play at the same time? In this example, it doesn't make sense for both to control the synthesiser at the same time, so we set IMPS to run in "call and response" mode, so that it only makes predictions when the human has stopped performing. We could also set up our mapping patch to use prediction messages for a different synth and use one of the simultaneous performance modes of IMPS.

### 2. Log some training data

You use the `predictive_music_model` command to log training data. If your interface has N inputs the dimension is N+1:

    python predictive_music_model.py --dimension=(N+1) --log

This command creates files in the `logs` directory with data like this:

    2019-01-17T12:37:38.109979,interface,0.3359375,0.296875,0.5078125
    2019-01-17T12:37:38.137938,interface,0.359375,0.296875,0.53125
    2019-01-17T12:37:38.160842,interface,0.375,0.3046875,0.1953125

These CSV files have the format: timestamp, source of message (interface or rnn), x_1, x_2, ...,  x_N.

You can log training data without using the RNN with the `o` switch (user only) if you like, or use a partially trained RNN and then collect more data.

    python predictive_music_model.py --dimension=(N+1) --log -o

Every time you run the `predictive_music_model`, a new log file is created so that you can build up a significant dataset!

### 3. Train an MDRNN

There's two steps for training: Generate a dataset file, and train the predictive model.

Use the `generate_dataset.py` command:

    python generate_dataset.py --dimension=(N+1)

This command collates all logs of dimension N+1 from the logs directory and saves the data in a compressed `.npz` file in the datasets directory. It will also print out some information about your dataset, in particular the total number of individual interactions. To have a useful dataset, it's good to start with more than 10,000 individual interactions but YMMV.

To train the model, use the `train_predictive_music_model.py` command---this can take a while on a normal computer, so be prepared to let your computer sit and think for a few hours! You'll have to decide what _size_ model to try to train: `xs`, `s`, `m`, `l`, `xl`. The size refers to the number of LSTM units in each layer of your model and roughly corresponds to "learning capacity" at a cost of slower training and predictions.
It's a good idea to start with an `xs` or `s` model, and the larger models are more relevant for quite large datasets (e.g., >1M individual interactions).

    python train_predictive_music_model.py --dimension=(N+1) --modelsize=xs --earlystopping

It's a good idea to use the "earlystopping" parameter to stop training after the model stops improving for 10 epochs.

### 4. Perform with your predictive model

Now that you have a trained model, you can run this command to start making predictions:

    python predictive_music_model.py -d=(N+1) --modelsize=xs --log

The `--log` switch logs all of your interactions as well as predictions for later re-training. (The dataset generator filters out RNN records so that you only train on human sourced data).

PS: the three scripts respond to the `--help` switch to show command line options. If there's something not documented or working, it would be great if you add an issue above to let me know or get in touch on Twitter at [@cpmpercussion](https://twitter.com/cpmpercussion).

## More about Mixture Density Recurrent Neural Networks

IMPS uses a mixture density recurrent neural network MDRNN to make predictions. This machine learning architecture is set up to predict the next in a sequence of multi-valued elements. The recurrent neural network uses LSTM units to remember information about past inputs and use this to help make decisions. The mixture density model at the end of the network allows continuous multi-valued elements to be sampled from a rich probability distribution. 

The network is illustrated here---every time IMPS receives an interaction message from your interface, it is sent to thorugh the LSTM layers to produce the parameters of a Gaussian mixture model. The predicted next interaction is sampled from this probability model.

![A Musical MDRNN](https://github.com/cpmpercussion/imps/raw/master/images/mdn_diagram.png)

The MDRNN is written in Keras and uses the [keras-mdn-layer](https://github.com/cpmpercussion/keras-mdn-layer) package. There's more info and tutorials about MDNs on [that github repo](https://github.com/cpmpercussion/keras-mdn-layer).
