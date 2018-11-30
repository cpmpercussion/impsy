"""
EMPI MDRNN Model.
Charles P. Martin, 2018
University of Oslo, Norway.
"""
import numpy as np
import pandas as pd
import mdn
import keras
import tensorflow as tf
import time
from .sample_data import *


tf.logging.set_verbosity(tf.logging.INFO)  # set logging.
NET_MODE_TRAIN = 'train'
NET_MODE_RUN = 'run'
MDN_MODEL_TENSORFLOW = 'tf'
MDN_MODEL_SKETCH = 'sketch'
MODEL_DIR = "./"
LOG_PATH = "./output-logs/"


def proc_generated_touch(touch):
    """ Processes a generated touch to have dt > 0, and 0 <= x <= 1 """
    dt = max(touch[0], 0.000454)
    x_loc = min(max(touch[1], 0), 1)
    return np.array([dt, x_loc])


def build_model(seq_len=30, hidden_units=256, num_mixtures=5, layers=2,
                     time_dist=True, inference=False, compile_model=True,
                     print_summary=True):
    """Builds a EMPI MDRNN model for training or inference.

    Keyword Arguments:
    seq_len : sequence length to unroll
    hidden_units : number of LSTM units in each layer
    num_mixtures : number of mixture components (5-10 is good)
    layers : number of layers (2 is good)
    time_dist : time distributed or not (default True)
    inference : inference network or training (default False)
    compile_model : compiles the model (default True)
    print_summary : print summary after creating mdoe (default True)
    """
    print("Building EMPI Model...")
    out_dim = 2
    # Set up training mode
    stateful = False
    batch_shape = None
    # Set up inference mode.
    if inference:
        stateful = True
        batch_shape = (1, 1, out_dim)
    inputs = keras.layers.Input(shape=(seq_len, out_dim), name='inputs',
                                batch_shape=batch_shape)
    lstm_in = inputs  # starter input for lstm
    for layer_i in range(layers):
        ret_seq = True
        if (layer_i == layers - 1) and not time_dist:
            # return sequences false if last layer, and not time distributed.
            ret_seq = False
        lstm_out = keras.layers.LSTM(hidden_units, name='lstm'+str(layer_i),
                                     return_sequences=ret_seq,
                                     stateful=stateful)(lstm_in)
        lstm_in = lstm_out

    mdn_layer = mdn.MDN(out_dim, num_mixtures, name='mdn_outputs')
    if time_dist:
        mdn_layer = keras.layers.TimeDistributed(mdn_layer, name='td_mdn')
    mdn_out = mdn_layer(lstm_out)  # apply mdn
    model = keras.models.Model(inputs=inputs, outputs=mdn_out)

    if compile_model:
        loss_func = mdn.get_mixture_loss_func(out_dim, num_mixtures)
        optimizer = keras.optimizers.Adam()
        # keras.optimizers.Adam(lr=0.0001))
        model.compile(loss=loss_func, optimizer=optimizer)

    model.summary()
    return model


def load_inference_model(model_file="", layers=2, units=512, mixtures=5, predict_moving=False):
    """Returns a Keras RoboJam model loaded from a file"""
    # TODO: make this parse the name to get the hyperparameters.
    # Decoding Model
    decoder = decoder = build_model(seq_len=1, hidden_units=units, num_mixtures=mixtures, layers=layers, time_dist=False, inference=True, compile_model=False, print_summary=True, predict_moving=predict_moving)
    decoder.load_weights(model_file)
    return decoder


# Performance Helper Functions
SCALE_FACTOR = 10  # scales input and output from the model. Should be the same between training and inference.


def generate_performance(model, n_mixtures, first_sample, time_limit=5.0, steps_limit=1000, temp=1.0, sigma_temp=0.0):
    """Generates a tiny performance up to 5 seconds in length."""
    out_dim = 2
    time = 0
    steps = 0
    prev_sample = first_sample
    print(prev_sample)
    performance = [prev_sample.reshape((out_dim,))]
    while (steps < steps_limit):  #  and time < time_limit
        params = model.predict(prev_sample.reshape(1,1,out_dim) * SCALE_FACTOR)
        prev_sample = mdn.sample_from_output(params[0], out_dim, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
        output_touch = prev_sample.reshape(out_dim,)
        # output_touch = constrain_touch(output_touch)
        performance.append(output_touch.reshape((out_dim,)))
        steps += 1
        time += output_touch[0]
    return np.array(performance)

def generate_sample(model, n_mixtures, prev_sample, temp=1.0, sigma_temp=0.0):
    """Generate prediction for a single touch."""
    out_dim = 2
    params = model.predict(prev_sample.reshape(1,1,out_dim) * SCALE_FACTOR)
    new_sample = mdn.sample_from_output(params[0], out_dim, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
    new_sample = new_sample.reshape(out_dim,)
    return new_sample

def random_sample():
    return np.array([(0.01 + (np.random.rand()-0.5)*0.005), (np.random.rand()-0.5)*2])


# def condition_and_generate(model, perf, n_mixtures, time_limit=5.0, steps_limit=1000, temp=1.0, sigma_temp=0.0, predict_moving=False):
#     """Conditions the network on an existing tiny performance, then generates a new one."""
#     if predict_moving:
#         out_dim = 4
#     else:
#         out_dim = 3
#     time = 0
#     steps = 0
#     # condition
#     for touch in perf:
#         params = model.predict(touch.reshape(1, 1, out_dim) * SCALE_FACTOR)
#         previous_touch = mdn.sample_from_output(params[0], out_dim, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
#         output = [previous_touch.reshape((out_dim,))]
#     # generate
#     while (steps < steps_limit and time < time_limit):
#         params = model.predict(previous_touch.reshape(1, 1, out_dim) * SCALE_FACTOR)
#         previous_touch = mdn.sample_from_output(params[0], out_dim, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
#         output_touch = previous_touch.reshape(out_dim,)
#         output_touch = constrain_touch(output_touch, with_moving=predict_moving)
#         output.append(output_touch.reshape((out_dim,)))
#         steps += 1
#         time += output_touch[2]
#     net_output = np.array(output)
#     return net_output

class MixtureRNN(object):
    """Mixture Density Network RNN using the SketchRNN's hand-coded loss function for the mixture of 2D Normals."""

    def __init__(self, mode=NET_MODE_TRAIN, n_hidden_units=128, n_mixtures=5, batch_size=100, sequence_length=120, layers=1):
        """Initialise the TinyJamNet model. Use mode='run' for evaluation graph and mode='train' for training graph."""
        # hyperparameters
        self.mode = mode
        self.n_hidden_units = n_hidden_units
        self.n_rnn_layers = layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.n_mixtures = n_mixtures  # number of mixtures
        self.n_input_units = 2  # Number of dimensions of the input (and sampled output) data
        self.lr = 1e-4  # could be 1e-3
        self.grad_clip = 1.0
        self.state = None
        self.use_input_dropout = False
        if self.mode is NET_MODE_TRAIN:
            self.model = build_model(seq_len=self.sequence_length, hidden_units=self.n_hidden_units, num_mixtures=self.n_mixtures, layers=self.n_rnn_layers,
                     time_dist=True, inference=False, compile_model=True, print_summary=True)
        else:
            self.model = build_model(seq_len=1, hidden_units=self.n_hidden_units, num_mixtures=self.n_mixtures, layers=self.n_rnn_layers,
                     time_dist=False, inference=True, compile_model=False, print_summary=True)

        self.run_name = self.get_run_name()

    def model_name(self):
        """Returns the name of the present model for saving to disk"""
        return "empi-rnn-" + str(self.n_rnn_layers) + "layers-" + str(self.n_hidden_units) + "units"

    def get_run_name(self):
        out = self.model_name() + "-"
        out += time.strftime("%Y%m%d-%H%M%S")
        return out

    # def train(self, data_manager, num_epochs, saving=True):
    #     """Train the network for the a number of epochs."""

    # def prepare_model_for_running(self, sess):
    #     """Load trained model and reset RNN state."""

    # def generate_touch(self, prev_touch, sess):
    #     """Generate prediction for a single touch."""
    #     input_touch = prev_touch.reshape([1, 1, self.n_input_units])  # Give input correct shape for one-at-a-time evaluation.
    #     if self.state is not None:
    #         feed = {self.x: input_touch, self.init_state: self.state}
    #     else:
    #         feed = {self.x: input_touch}
    #     pis, locs_1, locs_2, scales_1, scales_2, corr, self.state = sess.run([self.pis, self.locs_1, self.locs_2, self.scales_1, self.scales_2, self.corr, self.final_state], feed_dict=feed)
    #     x_1, x_2 = mixture_2d_normals.sample_mixture_model(pis[0], locs_1[0], locs_2[0], scales_1[0], scales_2[0], corr[0], temp=1.0, greedy=False)
    #     return np.array([x_1, x_2])

    # def generate_performance(self, first_touch, number, sess):
    #     self.prepare_model_for_running(sess)
    #     previous_touch = first_touch
    #     performance = [previous_touch.reshape((self.n_input_units,))]
    #     for i in range(number):
    #         previous_touch = self.generate_touch(previous_touch, sess)
    #         previous_touch = proc_generated_touch(previous_touch)
    #         performance.append(previous_touch.reshape((self.n_input_units,)))
    #     return np.array(performance)
