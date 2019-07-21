#!/usr/bin/python
import random
import numpy as np
import os
import argparse
import time
import datetime

# Hack to get openMP working annoyingly.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print("Script to train a predictive music interaction model.")

# Input and output to serial are bytes (0-255)
# Output to Pd is a float (0-1)
parser = argparse.ArgumentParser(description='Trains a predictive music interaction model')
parser.add_argument('-d', '--dimension', type=int, dest='dimension', default=4,
                    help='The dimension of the data to model, must be >= 2.')
parser.add_argument('-s', '--source', dest='sourcedir', default='logs',
                    help='The source directory to obtain .log files')
parser.add_argument("--modelsize", default="s", help="The model size: xs, s, m, l, xl", type=str)
parser.add_argument('-e', "--earlystopping", dest='earlystopping', action="store_true", help="Use early stopping")
parser.add_argument('-p', "--patience", type=int, dest='patience', default=10, help="The number of epochs patience for early stopping.")
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
if args.modelsize == 'xxs':
    mdrnn_units = 16
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 'xs':
    mdrnn_units = 32
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 's':
    mdrnn_units = 64
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 'm':
    mdrnn_units = 128
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 'l':
    mdrnn_units = 256
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 'xl':
    mdrnn_units = 512
    mdrnn_mixes = 5
    mdrnn_layers = 3
else:
    mdrnn_units = 128
    mdrnn_mixes = 5
    mdrnn_layers = 2

print("Model size:", args.modelsize)
print("Units:", mdrnn_units)
print("Layers:", mdrnn_layers)
print("Mixtures:", mdrnn_mixes)

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
dataset_location = 'datasets/'
dataset_filename = 'training-dataset-' + str(args.dimension) + 'd.npz'

with np.load(dataset_location + dataset_filename, allow_pickle=True) as loaded:
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
    slices += empi_mdrnn.slice_sequence_examples(seq,
                                                 SEQ_LEN+1,
                                                 step_size=SEQ_STEP)
X, y = empi_mdrnn.seq_to_overlapping_format(slices)
X = np.array(X) * empi_mdrnn.SCALE_FACTOR
y = np.array(y) * empi_mdrnn.SCALE_FACTOR

print("Number of training examples:")
print("X:", X.shape)
print("y:", y.shape)

# Setup Training Model
model = empi_mdrnn.build_model(seq_len=SEQ_LEN,
                               hidden_units=mdrnn_units,
                               num_mixtures=mdrnn_mixes,
                               layers=mdrnn_layers,
                               out_dim=args.dimension,
                               time_dist=TIME_DIST,
                               inference=False,
                               compile_model=True,
                               print_summary=True)

model_dir = "models/"
model_name = "musicMDRNN" + "-dim" + str(args.dimension) + "-layers" + str(mdrnn_layers) + "-units" + str(mdrnn_units) + "-mixtures" + str(mdrnn_mixes) + "-scale" + str(empi_mdrnn.SCALE_FACTOR)
date_string = datetime.datetime.today().strftime('%Y%m%d-%H_%M_%S')

filepath = model_dir + model_name + "-E{epoch:02d}-VL{val_loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='min')
terminateOnNaN = keras.callbacks.TerminateOnNaN()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
tboard = keras.callbacks.TensorBoard(log_dir='./logs/' + date_string + model_name,
                                     histogram_freq=2,
                                     batch_size=32,
                                     write_graph=True,
                                     update_freq='epoch')

callbacks = [checkpoint, terminateOnNaN, tboard]
if args.earlystopping:
    print("Enabling Early Stopping.")
    callbacks.append(early_stopping)
# Train
history = model.fit(X, y, batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=VAL_SPLIT,
                    callbacks=callbacks)

# Save final Model
model.save_weights(model_dir + model_name + ".h5")

print("Done, bye.")
