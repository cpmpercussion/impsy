# IMPSY: The Interactive Musical Predictive System

![MIT License](https://img.shields.io/github/license/cpmpercussion/keras-mdn-layer.svg?style=flat)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2580176.svg)](https://doi.org/10.5281/zenodo.2580176)
[![Install and run IMPSY](https://github.com/cpmpercussion/impsy/actions/workflows/python-app.yml/badge.svg)](https://github.com/cpmpercussion/impsy/actions/workflows/python-app.yml)
[![Coverage Status](https://coveralls.io/repos/github/cpmpercussion/impsy/badge.svg)](https://coveralls.io/github/cpmpercussion/impsy)
[![PyPI - Version](https://img.shields.io/pypi/v/IMPSY)](https://pypi.org/project/impsy/)

![Predictive Musical Interaction](https://github.com/cpmpercussion/impsy/raw/main/images/predictive_interaction.png)

IMPSY is a system for predicting musical control data in live performance. It uses a mixture density recurrent neural network (MDRNN) to observe control inputs over multiple time steps, predicting the next value of each step, and the time that expects the next value to occur. It provides an input and output interface over OSC and can work with musical interfaces with any number of real valued inputs (we've tried from 1-8). Several interactive paradigms are supported for call-response improvisation, as well as independent operation, and "filtering" of the performer's input. Whenever you use IMPSY, your input data is logged to build up a training corpus and a script is provided to train new versions of your model.

Here's a [demonstration video showing how IMPSY can be used with different musical interfaces:](https://www.youtube.com/embed/Kdmhrp2dfHw)

## Installation

IMPSY is written in Python with Keras and TensorFlow Probability, so it should work on any platform where Tensorflow can be installed. Python 3 is required and we use [Poetry](https://python-poetry.org) for managing dependencies. IMPSY currently relies on Python 3.11, TensorFlow 2.15.0, TensorFlow Probability 0.23.0, and keras-mdn-layer 0.3.0. You can see the dependencies in `pyproject.toml`.

To install IMPSY, first **ensure that you have a Python 3.11** installation available, you might want to use [pyenv](https://github.com/pyenv/pyenv) to manage different Python versions. Then you need to install [Poetry](https://python-poetry.org). The poetry install instructions vary depending on your preferences for a python setup this is likely to work on Linux, macOS or Windows (WSL):

    curl -sSL https://install.python-poetry.org | python3 -

Then you should clone this repository or download it to your computer:

    git clone https://github.com/cpmpercussion/impsy.git
    cd impsy

Then you can install the dependencies using Poetry:

    poetry install

Finally, you can test that IMPSY works:

    poetry run ./start_impsy.py --help

## How to use

There are four steps for using IMPSY. First, you'll need to setup your musical interface to send it OSC data and receive predictions the same way. Then you can log data, train the MDRNN, and make predictions using our provided scripts.

### 1. Connect music interface and synthesis software and configure IMPSY

IMPSY doesn't make any sound, it communicates with other sound making hardware or software and controller hardware or software via MIDI, OSC, WebSockets or serial. You could send and receive predictions from the same sofware (e.g., Pd with OSC input and output) or hardware (e.g., Arturia Microfreak with MIDI in and out). You can also have separate sources for input and output (input from an X-Touch controller with output to Max via OSC) and multiple sources at the same time.

You need to decide on a fixed number of inputs (or dimension) for your predictive model. This is the number of continuous outputs from your interface plus one (for time). So for an interface with 8 faders, the dimension will be 9. 

Your impsy configuration goes in a `.toml` file which by default is called `config.toml`. You can look in the `configs` directory to see many options including `default.toml` which has every possible section filled in.

For MIDI communication, IMPSY receives and sends message for one different note channel or CC for each dimension. Have a look at the `midi` block in `default.toml` for an example.

For OSC and Serial communication, IMPSY receives and sends on every dimension together in single dense messages. The messages to IMPSY should have the OSC address `/interface`, and then a float between 0 and 1 for each continuous output on your interface, e.g.:

    /interface 0 0.5 0.23 0.87 0.9 0.7 0.45 0.654

Your synthesiser software or interface needs to listen for messages from the IMPSY system as well. These have the same format with the OSC address `/prediction`. You can interpret these as interactions predicted to occur right when the message is sent.
The address and port of IMPSY's OSC server is configurable in the `osc` block, see `default.toml`.

Here's an example diagram for our 8-controller example, the [xtouch mini controller](https://www.musictribe.com/Categories/Behringer/Computer-Audio/Desktop-Controllers/X-TOUCH-MINI/p/P0B3M).

![Predictive Musical Interaction](https://github.com/cpmpercussion/impsy/raw/main/images/IMPS_connection_example.png)

In this example we've used Pd to connect the xtouch mini to IMPSY and to synthesis sounds. Our Pd mapping patch takes data from the xtouch and sends `/interface` OSC messages to IMPSY, it also receives `/prediction` OSC message back from IMPSY whenever they occur. Of course, whenever the user performs with the controller, the mapping patch sends commands to the synthesiser patch to make sound. Whenever `/prediction` messages are received, these also trigger changes in the synth patch, and we also send MIDI messages back to the xtouch controller to update its lights so that the performer knows what IMPSY is predicting.

So what happens if IMPSY and the performer play at the same time? In this example, it doesn't make sense for both to control the synthesiser at the same time, so we set IMPSY to run in "call and response" mode, so that it only makes predictions when the human has stopped performing. We could also set up our mapping patch to use prediction messages for a different synth and use one of the simultaneous performance modes of IMPS.

### 2. Log some training data

You use the `run` command to log training data. If your interface has N inputs the dimension is N+1:

    poetry run ./start_impsy run

This command creates files in the `logs` directory with data like this:

    2019-01-17T12:37:38.109979,interface,0.3359375,0.296875,0.5078125
    2019-01-17T12:37:38.137938,interface,0.359375,0.296875,0.53125
    2019-01-17T12:37:38.160842,interface,0.375,0.3046875,0.1953125

These CSV files have the format: timestamp, source of message (interface or rnn), x_1, x_2, ...,  x_N.

You can log training data without using the RNN with the `useronly` mode in `config.toml`, make sure the `interaction` block has:
```
mode = "useronly"
```
Look at `configs/user-only-example.toml` for an example.

Every time you use IMPS' "run" command, a new log file is created so that you can build up a significant dataset!

### 3. Train an MDRNN

There's two steps for training: Generate a dataset file, and train the predictive model.

Use the `dataset` command:

    poetry run ./start_impsy dataset --dimension (N+1)

This command collates all logs of dimension N+1 from the logs directory and saves the data in a compressed `.npz` file in the datasets directory. It will also print out some information about your dataset, in particular the total number of individual interactions. To have a useful dataset, it's good to start with more than 10,000 individual interactions but YMMV.

To train the model, use the `train` command---this can take a while on a normal computer, so be prepared to let your computer sit and think for a few hours! You'll have to decide what _size_ model to try to train: `xs`, `s`, `m`, `l`, `xl`. The size refers to the number of LSTM units in each layer of your model and roughly corresponds to "learning capacity" at a cost of slower training and predictions.
It's a good idea to start with an `xs` or `s` model, and the larger models may work better for quite large datasets (e.g., >1M individual interactions).

    poetry run ./start_impsy train --dimension (N+1) --modelsize s

It's a good idea to use the `earlystopping` option to stop training after the model stops improving for 10 epochs.

By default, your trained model will be saved in the `models` directory in `.keras` and `.tflite` format.

> WHat's with `.keras` and `.tflite` files? Both Keras and TFLite files have all the information needed to reconstruct a trained IMPSY neural network. `.keras` is the Keras machine learning framework's native format and `.tflite` is TensorFlow Lite's optimised model format. Until 2024 we used Keras' native model storage but Tensorflow Lite turns out to be more than 20x faster so it's almost always a better idea to use the `.tflite` file.

### 4. Perform with your predictive model

Now that you have a trained model, make sure that it is listed in your `config.toml` file, for example under `model` you might list:
```
dimension = 9
file = "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
size = "s"
```
Which will load a 9d TFLite model of the "small" size. You can run this command to start making predictions:

    poetry run ./start_impsy run 

PS: all the IMPSY commands respond to the `--help` switch to show command line options. If there's something not documented or working, it would be great if you add an issue above to let me know.

### Using Docker to run IMPSy

We provide the docker image [`charlepm/impsy`](https://hub.docker.com/r/charlepm/impsy) which includes IMPSY with Poetry and required libraries installed.

You can use the docker image to try out impsy or even use it in production if you are using OSC communication to a sound source or musical interface. MIDI doesn't work in the docker container (not sure how this could be achieved but if someone has a good idea...). The docker container is defined at `examples/Dockerfile`.

We also have a docker compose file to start IMPSY as well as the web user interface: `docker-compose.yml`

#### Docker compose configuration

From the IMPSY main directory run:
```
docker compose -f docker-compose.yml up
```
Then you can navigate to `http://127.0.0.1:4000` to view the web interface. OSC communication happens through ports 6000 and 6001. The local `config.toml`, and `datasets`, `logs` and `models` directories are mapped into the docker containers.

You will need a special client address, `host.docker.internal` to send messages out of the docker containers to your host computer. Make sure this is listed under `client_ip` in `config.toml`. See `examples/pd-workshop-example.toml` for an example.

#### Using docker to create datasets or train models

You can run a docker container and use different impsy commands from the command line as well:

```
docker run -d -v $(pwd)/datasets:/code/datasets -v $(pwd)/logs:/code/logs -v $(pwd)/models:/code/models -v $(pwd)/config.toml:/code/config.toml charlepm/impsy poetry run ./start_impsy.py --help
```
This can be useful to use the `dataset` or `train` commands to generate new datasets and models.


## More about Mixture Density Recurrent Neural Networks

IMPSY uses a mixture density recurrent neural network MDRNN to make predictions. This machine learning architecture is set up to predict the next in a sequence of multi-valued elements. The recurrent neural network uses LSTM units to remember information about past inputs and use this to help make decisions. The mixture density model at the end of the network allows continuous multi-valued elements to be sampled from a rich probability distribution. 

The network is illustrated here---every time IMPSY receives an interaction message from your interface, it is sent to thorugh the LSTM layers to produce the parameters of a Gaussian mixture model. The predicted next interaction is sampled from this probability model.

![A Musical MDRNN](https://github.com/cpmpercussion/impsy/raw/main/images/mdn_diagram.png)

The MDRNN is written in Keras and uses the [keras-mdn-layer](https://github.com/cpmpercussion/keras-mdn-layer) package. There's more info and tutorials about MDNs on [that github repo](https://github.com/cpmpercussion/keras-mdn-layer).
