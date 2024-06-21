"""
EMPI MDRNN Model.
Charles P. Martin, 2018
University of Oslo, Norway.
"""

import numpy as np
import tensorflow as tf
import keras_mdn_layer as mdn
import time

NET_MODE_TRAIN = "train"
NET_MODE_RUN = "run"
MODEL_DIR = "./models/"
LOG_PATH = "./logs/"
SCALE_FACTOR = 10  # scales input and output from the model. Should be the same between training and inference.


# Functions for slicing up data
def slice_sequence_examples(sequence, num_steps, step_size=1):
    """Slices a sequence into examples of length
    num_steps with step size step_size."""
    xs = []
    for i in range((len(sequence) - num_steps) // step_size + 1):
        example = sequence[(i * step_size) : (i * step_size) + num_steps]
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
    """Return the examples in seq to singleton format."""
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs, ys)


def build_model(
    seq_len=30,
    hidden_units=256,
    num_mixtures=5,
    layers=2,
    out_dim=2,
    time_dist=True,
    inference=False,
    print_summary=True,
):
    """Builds a EMPI MDRNN model for training or inference.

    Keyword Arguments:
    seq_len : sequence length to unroll
    hidden_units : number of LSTM units in each layer
    num_mixtures : number of mixture components (5-10 is good)
    layers : number of layers (2 is good)
    out_dim : number of dimensions for the model = number of degrees of freedom + 1 (time)
    time_dist : time distributed or not (default True)
    inference : inference network or training (default False)
    print_summary : print summary after creating mode (default True)
    """
    print("Building EMPI Model...")
    if inference:
        state_input_output = True
    else:
        state_input_output = False
    data_input = tf.keras.layers.Input(
        shape=(seq_len, out_dim), name="inputs"
    )
    lstm_in = data_input  # starter input for lstm
    state_inputs = [] # storage for LSTM state inputs
    state_outputs = [] # storage for LSTM state outputs

    for layer_i in range(layers):
        return_sequences = True
        if (layer_i == layers - 1) and not time_dist:
            # return sequences false if last layer, and not time distributed.
            return_sequences = False
        state_input = None
        if state_input_output:
            state_h_input = tf.keras.layers.Input(shape=(hidden_units,), name=f"state_h_{layer_i}")
            state_c_input = tf.keras.layers.Input(shape=(hidden_units,), name=f"state_c_{layer_i}")
            state_input = [state_h_input, state_c_input]
            state_inputs += state_input
        lstm_out, state_h_output, state_c_output = tf.keras.layers.LSTM(
            hidden_units,
            name=f"lstm_{layer_i}",
            return_sequences=return_sequences,
            return_state= True # state_input_output # better to keep these outputs and just not use.
        )(lstm_in, initial_state=state_input)
        lstm_in = lstm_out
        state_outputs += [state_h_output, state_c_output]

    mdn_layer = mdn.MDN(out_dim, num_mixtures, name="mdn_outputs")
    if time_dist:
        mdn_layer = tf.keras.layers.TimeDistributed(mdn_layer, name="td_mdn")
    mdn_out = mdn_layer(lstm_out)  # apply mdn
    if inference:
        # for inference, need to track state of the model
        inputs = [data_input] + state_inputs
        outputs = [mdn_out] + state_outputs
    else:
        # for training we don't need to keep track of state in the model
        inputs = data_input
        outputs = mdn_out
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    if not inference:
        # only need loss function and compile when training
        loss_func = mdn.get_mixture_loss_func(out_dim, num_mixtures)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(loss=loss_func, optimizer=optimizer)

    model.summary()
    return model


def load_inference_model(
    model_file="", layers=2, units=512, mixtures=5, predict_moving=False
):
    """Returns an IMPS model loaded from a file"""
    # TODO: make this parse the name to get the hyperparameters.
    decoder = decoder = build_model(
        seq_len=1,
        hidden_units=units,
        num_mixtures=mixtures,
        layers=layers,
        time_dist=False,
        inference=True,
        print_summary=True,
        predict_moving=predict_moving,
    )
    decoder.load_weights(model_file)
    return decoder


def random_sample(out_dim=2):
    """Generate a random sample in format (dt, x_1, ..., x_n), where dt is positive
    and the x_i are between 0 and 1."""
    output = np.random.rand(out_dim)
    output[0] = (
        0.01 + (np.random.rand() - 0.5) * 0.005
    )  # TODO: see if this dt heuristic should change
    return output


def proc_generated_touch(x_input, out_dim=2):
    """Processes a generated touch in the format (dt, x)
    such that dt > 0, and 0 <= x <= 1"""
    dt = np.maximum(
        x_input[0], 0.000454
    )  # TODO: see if the min value of dt should change.
    x_output = np.minimum(np.maximum(x_input[1:], 0), 1)
    return np.concatenate([np.array([dt]), x_output])


class PredictiveMusicMDRNN(object):
    """EMPI MDRNN object for convenience in the run script."""

    def __init__(
        self,
        mode=NET_MODE_TRAIN,
        dimension=2,
        n_hidden_units=128,
        n_mixtures=5,
        batch_size=100,
        sequence_length=120,
        layers=2,
    ):
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
        # self.name="impsy-mdrnn"


        if self.mode is NET_MODE_TRAIN:
            self.model = build_model(
                seq_len=self.sequence_length,
                hidden_units=self.n_hidden_units,
                num_mixtures=self.n_mixtures,
                layers=self.n_rnn_layers,
                out_dim=self.dimension,
                time_dist=True,
                inference=False,
                print_summary=True,
            )
        else:
            self.model = build_model(
                seq_len=1,
                hidden_units=self.n_hidden_units,
                num_mixtures=self.n_mixtures,
                layers=self.n_rnn_layers,
                out_dim=self.dimension,
                time_dist=False,
                inference=True,
                print_summary=True,
            )

        self.run_name = self.get_run_name()
        self.reset_lstm_states()


    def reset_lstm_states(self):
        states = []
        for i in range(self.n_rnn_layers):
            states += [np.zeros((1,self.n_hidden_units), dtype=np.float32), np.zeros((1,self.n_hidden_units), dtype=np.float32)]
        assert len(states) == self.n_rnn_layers * 2, "length of states list needs to be RNN layers times 2 (h and c for each)"
        self.lstm_states = states

    def model_name(self):
        """Returns the name of the present model for saving to disk"""
        return (
            "musicMDRNN"
            + "-dim"
            + str(self.dimension)
            + "-layers"
            + str(self.n_rnn_layers)
            + "-units"
            + str(self.n_hidden_units)
            + "-mixtures"
            + str(self.n_mixtures)
            + "-scale"
            + str(SCALE_FACTOR)
        )

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
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )
        terminateOnNaN = tf.keras.callbacks.TerminateOnNaN()
        tboard = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_PATH + self.run_name,
            histogram_freq=2,
            batch_size=32,
            write_graph=True,
            update_freq="epoch",
        )
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
        history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=num_epochs,
            validation_split=self.val_split,
            callbacks=callbacks,
        )
        return history

    def prepare_model_for_running(self):
        """Reset RNN state."""
        self.reset_lstm_states()  # reset LSTM state.

    def generate_touch(self, prev_sample):
        """Generate one forward prediction from a previous sample in format
        (dt, x_1,...,x_n). Pi and Sigma temperature are adjustable."""
        assert len(prev_sample) == self.dimension, "Only works with samples of the same dimension as the network"
        # print("Input sample", prev_sample)
        input_list = [prev_sample.reshape(1, 1, self.dimension) * SCALE_FACTOR] + self.lstm_states
        model_output = self.model(input_list)
        mdn_params = model_output[0][0].numpy()
        self.lstm_states = model_output[1:] # update storage of LSTM state

        # sample from the MDN:
        new_sample = (
            mdn.sample_from_output(
                mdn_params, self.dimension, self.n_mixtures, temp=self.pi_temp, sigma_temp=self.sigma_temp
            )
            / SCALE_FACTOR
        )
        new_sample = new_sample.reshape(
            self.dimension,
        )
        return new_sample
