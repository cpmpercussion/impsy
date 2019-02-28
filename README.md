# IMPS: The Interactive Musical Predictive System

![Predictive Musical Interaction](https://github.com/cpmpercussion/imps/raw/master/images/predictive_interaction.png)

IMPS is a system for predicting musical control data in live performance. It uses a mixture density recurrent neural network (MDRNN) to observe control inputs over multiple time steps, predicting the next value of each step, and the time that expects the next value to occur. It provides an input and output interface over OSC and can work with musical interfaces with any number of real valued inputs (we've tried from 1-8). Several interactive paradigms are supported for call-response improvisation, as well as independent operation, and "filtering" of the performer's input. Whenever you use IMPS, your input data is logged to build up a training corpus and a script is provided to train new versions of your model.

## Installation

IMPS is written in Python with Keras and TensorFlow Probability, so it should work on any platform where Tensorflow can be installed. Python 3 is required. The python requirements can be installed as follows:

    pip install -r requirements.txt

The Raspberry Pi requires some care to install matching version of TensorFlow and TensorFlow Probability, so we have provided a special requirements file:

    pip install -r pi_requirements.txt

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

Use the `generate_dataset` command:

    python generate_dataset --dimension=(N+1)

This command collates all logs of dimension N+1 from the logs directory and saves the data in a compressed `.npz` file in the datasets directory. It will also print out some information about your dataset, in particular the total number of individual interactions. To have a useful dataset, it's good to start with more than 10,000 individual interactions but YMMV.

To train the model, use the `train_predictive_music_model` command---this can take a while on a normal computer, so be prepared to let your computer sit and think for a few hours! You'll have to decide what _size_ model to try to train: `xs`, `s`, `m`, `l`, `xl`. The size refers to the number of LSTM units in each layer of your model and roughly corresponds to "learning capacity" at a cost of slower training and predictions.
It's a good idea to start with an `xs` or `s` model, and the larger models are more relevant for quite large datasets (e.g., >1M individual interactions).

    python train_predictive_music_model.py --dimension=(N+1) --modelsize=xs --earlystopping

It's a good idea to use the "earlystopping" parameter to stop training after the model stops improving for 10 epochs.

### 4. Perform with your predictive model

Now that you have a trained model, you can run this command to start making predictions:

    python predictive_music_model.py -d=(N+1) --modelsize=xs --log

THe `--log` switch logs all of your interactions as well as predictions for later re-training. (The dataset generator filters out RNN records so that you only train on human sourced data).

## More about Mixture Density Recurrent Neural Networks

![A Musical MDRNN](https://github.com/cpmpercussion/imps/raw/master/images/mdn_diagram.png)

