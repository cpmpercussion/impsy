#!/usr/bin/python
import random
import numpy as np
import pandas as pd
import os
import argparse
import time


print("Script to train a predictive music interaction model.")

# Input and output to serial are bytes (0-255)
# Output to Pd is a float (0-1)
parser = argparse.ArgumentParser(description='Trains a predictive music interaction model')
parser.add_argument('-d', '--dimension', type=int, dest='dimension', default=2,
                    help='The dimension of the data to model, must be >= 2.')
parser.add_argument('-s', '--source', dest='sourcedir', default='logs',
                    help='The source directory to obtain .log files')
parser.add_argument("--modelsize", default="s", help="The model size: s, m, l, xl")
args = parser.parse_args()


# Import Keras
import empi_mdrnn
import keras
import keras.backend as K
import tensorflow as tf
# Set up environment.
# Only for GPU use:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Choose model parameters.
if args.modelsize is 's':
    mdrnn_units = 64
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize is 'm':
    mdrnn_units = 128
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize is 'l':
    mdrnn_units = 256
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize is 'xl':
    mdrnn_units = 512
    mdrnn_mixes = 5
    mdrnn_layers = 3
else:
    mdrnn_units = 128
    mdrnn_mixes = 5
    mdrnn_layers = 2

# Model Hyperparameters
SEQ_LEN = 50
SEQ_STEP = 1
TIME_DIST = True

# Training Hyperparameters:
BATCH_SIZE = 64
EPOCHS = 100
VAL_SPLIT = 0.10

# Set random seed for reproducibility
SEED = 2345
random.seed(SEED)
np.random.seed(SEED)

# Load dataset
# Load tiny performance data from compressed file.
dataset_location = '../datasets/'
dataset_filename = 'training-dataset-' + str(args.dimension) + 'd.npz'

with np.load(dataset_location + dataset_filename) as loaded:
    perfs = loaded['perfs']

print("Loaded perfs:", len(perfs))
print("Num touches:", np.sum([len(l) for l in perfs]))
corpus = perfs  # might need to do some processing here...processing
# Restrict corpus to sequences longer than the corpus.
corpus = [l for l in corpus if len(l) > SEQ_LEN+1]
print("Corpus Examples:", len(corpus))
# Prepare training data as X and Y.
slices = []
for seq in corpus:
    slices += empi_mdrnn.slice_sequence_examples(seq, SEQ_LEN+1, step_size=SEQ_STEP)
X, y = empi_mdrnn.seq_to_overlapping_format(slices)
X = np.array(X) * empi_mdrnn.SCALE_FACTOR
y = np.array(y) * empi_mdrnn.SCALE_FACTOR

print("Number of training examples:")
print("X:", X.shape)
print("y:", y.shape)

# Setup Training Model
model = empi_mdrnn.build_model(seq_len=SEQ_LEN, hidden_units=mdrnn_units, num_mixtures=mdrnn_mixes, layers=mdrnn_layers, time_dist=TIME_DIST, inference=False, compile_model=True, print_summary=True)

# Setup callbacks
model_path = "empi_mdrnn" + "-layers" + str(N_LAYERS) + "-units" + str(HIDDEN_UNITS) + "-mixtures" + str(NUMBER_MIXTURES) + "-scale" + str(empi_mdrnn.SCALE_FACTOR)
filepath = model_path + "-E{epoch:02d}-VL{val_loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
terminateOnNaN = keras.callbacks.TerminateOnNaN()
tboard = keras.callbacks.TensorBoard(log_dir='./logs/'+model_path, histogram_freq=2, batch_size=32, write_graph=True, update_freq='epoch')

# Train
history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT, callbacks=[checkpoint, terminateOnNaN, tboard])
#history = model.fit_generator(generator, steps_per_epoch=300, epochs=100, verbose=1, initial_epoch=0)

# Save final Model
model.save_weights('model_path' + '-final.hdf5')  # creates a HDF5 file of the model

print("Done, bye.")
