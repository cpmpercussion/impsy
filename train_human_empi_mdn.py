#!/usr/bin/python3
import random
import numpy as np
import pandas as pd
import time

import empi_mdrnn
import keras
import tensorflow as tf
import os

# Set up environment.
# Only for GPU use:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# Model Hyperparameters
SEQ_LEN = 50
SEQ_STEP = 1
HIDDEN_UNITS = 128
N_LAYERS = 2
NUMBER_MIXTURES = 5
TIME_DIST = True

# Training Hyperparameters:
BATCH_SIZE = 64
EPOCHS = 100
VAL_SPLIT = 0.10

# Set random seed for reproducibility
SEED = 2345
random.seed(SEED)
np.random.seed(SEED)


# Import Human Data CSV and create dt 
perf_df = pd.read_csv('./data/2018-01-25T14-04-35-rnnbox.csv', header=0, index_col=0, parse_dates=['date'])
perf_df['time'] = perf_df.index
perf_df['seconds'] = perf_df.index
perf_df.time = perf_df.time.diff()
perf_df.time = perf_df.time.dt.total_seconds()
perf_df = perf_df.dropna()
perf_df.value = perf_df.value / 255.0
corpus_df = pd.DataFrame({'t': perf_df.time, 'x': perf_df.value})
corpus = np.array(corpus_df)
print("Shape of corpus array:", corpus.shape)
corpus_df.describe()
perf_df.seconds = perf_df.seconds - perf_df.seconds[0]
perf_df.seconds = perf_df.seconds.dt.total_seconds()
perf_df.describe()

slices = empi_mdrnn.slice_sequence_examples(corpus, SEQ_LEN+1, step_size=SEQ_STEP)

X, y = empi_mdrnn.seq_to_overlapping_format(slices)

X = np.array(X) * empi_mdrnn.SCALE_FACTOR
y = np.array(y) * empi_mdrnn.SCALE_FACTOR

print("Number of training examples:")
print("X:", X.shape)
print("y:", y.shape)

# Setup Training Model
model = empi_mdrnn.build_model(seq_len=SEQ_LEN, hidden_units=HIDDEN_UNITS, num_mixtures=NUMBER_MIXTURES, layers=2, time_dist=TIME_DIST, inference=False, compile_model=True, print_summary=True)

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
model.save('model_path' + '-final.hdf5')  # creates a HDF5 file of the model

print("Done, bye.")
