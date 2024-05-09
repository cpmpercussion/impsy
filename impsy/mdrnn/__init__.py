"""
EMPI MDRNN Model.
Charles P. Martin, 2018
University of Oslo, Norway.
"""
import numpy as np
import tensorflow.compat.v1 as tf
import keras_mdn_layer as mdn
import time

tf.logging.set_verbosity(tf.logging.INFO)  # set logging.
NET_MODE_TRAIN = 'train'
NET_MODE_RUN = 'run'
MODEL_DIR = "./models/"
LOG_PATH = "./logs/"
SCALE_FACTOR = 10  # scales input and output from the model. Should be the same between training and inference.


# Functions for slicing up data
def slice_sequence_examples(sequence, num_steps, step_size=1):
    """ Slices a sequence into examples of length
        num_steps with step size step_size."""
    xs = []
    for i in range((len(sequence) - num_steps) // step_size + 1):
        example = sequence[(i * step_size): (i * step_size) + num_steps]
        xs.append(example)
    return xs


def seq_to_overlapping_format(examples):
    """Takes sequences of seq_len+1 and returns overlapping
    sequences of seq_len."""
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[1:])
    return (xs, ys)


def seq_to_singleton_format(examples):
    """Return the examples in seq to singleton format.
    """
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs, ys)


def build_model(seq_len=30, hidden_units=256, num_mixtures=5, layers=2,
                out_dim=2, time_dist=True, inference=False, compile_model=True,
                print_summary=True):
    """Builds a EMPI MDRNN model for training or inference.

    Keyword Arguments:
    seq_len : sequence length to unroll
    hidden_units : number of LSTM units in each layer
    num_mixtures : number of mixture components (5-10 is good)
    layers : number of layers (2 is good)
    out_dim : number of dimensions for the model = number of degrees of freedom + 1 (time)
    time_dist : time distributed or not (default True)
    inference : inference network or training (default False)
    compile_model : compiles the model (default True)
    print_summary : print summary after creating mdoe (default True)
    """
    print("Building EMPI Model...")
    # Set up training mode
    stateful = False
    # batch_shape = None
    batch_size = None
    # Set up inference mode.
    if inference:
        stateful = True
        batch_size = 1
        #batch_shape = (1, 1, out_dim)
    inputs = tf.keras.layers.Input(shape=(seq_len, out_dim), name='inputs',
                                   batch_size=batch_size)
                                # batch_shape=batch_shape)
    lstm_in = inputs  # starter input for lstm
    for layer_i in range(layers):
        ret_seq = True
        if (layer_i == layers - 1) and not time_dist:
            # return sequences false if last layer, and not time distributed.
            ret_seq = False
        lstm_out = tf.keras.layers.LSTM(hidden_units, name='lstm'+str(layer_i),
                                     return_sequences=ret_seq,
                                     stateful=stateful)(lstm_in)
        lstm_in = lstm_out

    mdn_layer = mdn.MDN(out_dim, num_mixtures, name='mdn_outputs')
    if time_dist:
        mdn_layer = tf.keras.layers.TimeDistributed(mdn_layer, name='td_mdn')
    mdn_out = mdn_layer(lstm_out)  # apply mdn
    model = tf.keras.models.Model(inputs=inputs, outputs=mdn_out)

    if compile_model:
        loss_func = mdn.get_mixture_loss_func(out_dim, num_mixtures)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(loss=loss_func, optimizer=optimizer)

    model.summary()
    return model


def load_inference_model(model_file="", layers=2, units=512, mixtures=5, predict_moving=False):
    """Returns an IMPS model loaded from a file"""
    # TODO: make this parse the name to get the hyperparameters.
    decoder = decoder = build_model(seq_len=1, hidden_units=units, num_mixtures=mixtures, layers=layers, time_dist=False, inference=True, compile_model=False, print_summary=True, predict_moving=predict_moving)
    decoder.load_weights(model_file)
    return decoder


def random_sample(out_dim=2):
    """ Generate a random sample in format (dt, x_1, ..., x_n), where dt is positive
    and the x_i are between 0 and 1."""
    output = np.random.rand(out_dim)
    output[0] = (0.01 + (np.random.rand()-0.5)*0.005)  # TODO: see if this dt heuristic should change
    return output


def proc_generated_touch(x_input, out_dim=2):
    """ Processes a generated touch in the format (dt, x)
        such that dt > 0, and 0 <= x <= 1 """
    dt = np.maximum(x_input[0], 0.000454)  # TODO: see if the min value of dt shoud change.
    x_output = np.minimum(np.maximum(x_input[1:], 0), 1)
    return np.concatenate([np.array([dt]), x_output])


def generate_sample(model, n_mixtures, prev_sample, pi_temp=1.0, sigma_temp=0.0, out_dim=2):
    """Generate one forward prediction from a previous sample in format
    (dt, x_1,...,x_n). Pi and Sigma temperature are adjustable."""
    params = model.predict(prev_sample.reshape(1, 1, out_dim) * SCALE_FACTOR)
    new_sample = mdn.sample_from_output(params[0], out_dim, n_mixtures, temp=pi_temp, sigma_temp=sigma_temp) / SCALE_FACTOR
    new_sample = new_sample.reshape(out_dim,)
    return new_sample


def generate_performance(model, n_mixtures, first_sample, time_limit=None, steps_limit=1000, pi_temp=1.0, sigma_temp=0.0, out_dim=2):
    """Generates a performance of (dt, x) pairs, up to a step_limit.
    Time limit is not presently implemented.
    """
    time = 0
    steps = 0
    prev_sample = first_sample
    print(prev_sample)
    performance = [prev_sample.reshape((out_dim,))]
    while (steps < steps_limit):  # and time < time_limit
        params = model.predict(prev_sample.reshape(1, 1, out_dim) * SCALE_FACTOR)
        prev_sample = mdn.sample_from_output(params[0], out_dim, n_mixtures,
                                             temp=pi_temp,
                                             sigma_temp=sigma_temp)
        prev_sample = prev_sample / SCALE_FACTOR
        output_touch = prev_sample.reshape(out_dim,)
        output_touch = proc_generated_touch(output_touch)
        performance.append(output_touch.reshape((out_dim,)))
        steps += 1
        time += output_touch[0]
    return np.array(performance)


class PredictiveMusicMDRNN(object):
    """EMPI MDRNN object for convenience in the run script."""

    def __init__(self, mode=NET_MODE_TRAIN, dimension=2, n_hidden_units=128, n_mixtures=5, batch_size=100, sequence_length=120, layers=2):
        """Initialise the MDRNN model. Use mode='run' for evaluation graph and
        mode='train' for training graph."""
        # network parameters
        self.dimension = dimension
        self.mode = mode
        self.n_hidden_units = n_hidden_units
        self.n_rnn_layers = layers
        self.n_mixtures = n_mixtures  # number of mixtures
        # Training parameters
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.val_split = 0.10
        # Sampling hyperparameters
        self.pi_temp = 1.5
        self.sigma_temp = 0.01

        if self.mode is NET_MODE_TRAIN:
            self.model = build_model(seq_len=self.sequence_length,
                                     hidden_units=self.n_hidden_units,
                                     num_mixtures=self.n_mixtures,
                                     layers=self.n_rnn_layers,
                                     out_dim=self.dimension,
                                     time_dist=True,
                                     inference=False,
                                     compile_model=True,
                                     print_summary=True)
        else:
            self.model = build_model(seq_len=1,
                                     hidden_units=self.n_hidden_units,
                                     num_mixtures=self.n_mixtures,
                                     layers=self.n_rnn_layers,
                                     out_dim=self.dimension,
                                     time_dist=False,
                                     inference=True,
                                     compile_model=False,
                                     print_summary=True)

        self.run_name = self.get_run_name()

    def model_name(self):
        """Returns the name of the present model for saving to disk"""
        return "musicMDRNN" + "-dim" + str(self.dimension) + "-layers" + str(self.n_rnn_layers) + "-units" + str(self.n_hidden_units) + "-mixtures" + str(self.n_mixtures) + "-scale" + str(SCALE_FACTOR)

    def load_model(self, model_file=None):
        if model_file is None:
            model_file = MODEL_DIR + self.model_name() + ".h5"
        try:
            self.model.load_weights(model_file)
        except OSError as err:
            print("OS error: {0}".format(err))
            print("MDRNN could not be loaded from file:", model_file)
            print("MDRNN is untrained.")

    def get_run_name(self):
        out = self.model_name() + "-"
        out += time.strftime("%Y%m%d-%H%M%S")
        return out

    def train(self, X, y, num_epochs=10, saving=True):
        """Train the network for the a number of epochs."""
        # Setup callbacks
        filepath = MODEL_DIR + self.model_name() + "-E{epoch:02d}-VL{val_loss:.2f}.hdf5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        terminateOnNaN = tf.keras.callbacks.TerminateOnNaN()
        tboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH+self.run_name, histogram_freq=2, batch_size=32, write_graph=True, update_freq='epoch')
        callbacks = [terminateOnNaN, tboard]
        if saving:
            callbacks.append(checkpoint)

        # Do the data scaling in here.
        X = np.array(X) * SCALE_FACTOR
        y = np.array(y) * SCALE_FACTOR
        print("Training corpus has shape:")
        print("X:", X.shape)
        print("y:", y.shape)

        # Train
        history = self.model.fit(X, y, batch_size=self.batch_size,
                                 epochs=num_epochs,
                                 validation_split=self.val_split,
                                 callbacks=callbacks)
        return history

    def prepare_model_for_running(self):
        """Reset RNN state."""
        self.model.reset_states()  # reset LSTM state.

    def generate_touch(self, prev_sample):
        # TODO - do something with the session.
        output = generate_sample(self.model, self.n_mixtures, prev_sample,
                                 pi_temp=self.pi_temp,
                                 sigma_temp=self.sigma_temp,
                                 out_dim=self.dimension)
        return output

    def generate_performance(self, first_sample, number):
        return generate_performance(self.model, self.n_mixtures,
                                    first_sample, time_limit=None,
                                    steps_limit=number, pi_temp=self.pi_temp,
                                    sigma_temp=self.sigma_temp,
                                    out_dim=self.dimension)
