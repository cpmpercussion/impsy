"""
IMPSY MDRNN Model.
Charles P. Martin, 2018
University of Oslo, Norway.
"""

import numpy as np
import tensorflow as tf
import keras_mdn_layer as mdn
import time
import datetime

NET_MODE_TRAIN = "train"
NET_MODE_RUN = "run"
LOG_PATH = "./logs/"
SCALE_FACTOR = 10  # scales input and output from the model. Should be the same between training and inference.


# def load_inference_model(
#     model_file="", layers=2, units=512, mixtures=5, predict_moving=False
# ):
#     """Returns an IMPS model loaded from a file"""
#     # TODO: make this parse the name to get the hyperparameters.
#     decoder = build_model(
#         seq_len=1,
#         hidden_units=units,
#         num_mixtures=mixtures,
#         layers=layers,
#         time_dist=False,
#         inference=True,
#         print_summary=True,
#         predict_moving=predict_moving,
#     )
#     decoder.load_weights(model_file)
#     return decoder


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
    """Builds and operates a mixture density recurrent neural network model."""

    def __init__(
        self,
        mode=NET_MODE_TRAIN,
        dimension=2,
        n_hidden_units=128,
        n_mixtures=5,
        sequence_length=30,
        layers=2,
    ):
        """Initialise the MDRNN model. Use mode='run' for evaluation graph and
        mode='train' for training graph.

        Keyword Arguments:

        dimension : number of dimensions for the model = number of degrees of freedom + 1 (time)
        n_hidden_units : number of LSTM units in each layer
        n_mixtures : number of mixture components (5-10 is good)
        layers : number of layers (2 is good)
        seq_len : sequence length to unroll
        batch_size : size of batch for training (not used so far)
        """
        # network parameters
        self.dimension = dimension
        self.mode = mode
        self.n_hidden_units = n_hidden_units
        self.n_rnn_layers = layers
        self.n_mixtures = n_mixtures  # number of mixtures
        # Sampling hyperparameters
        self.pi_temp = 1.5
        self.sigma_temp = 0.01
        # self.name="impsy-mdrnn"
        if self.mode == NET_MODE_RUN:
            self.sequence_length = 1
            self.inference = True
            self.time_dist = False
        else:
            self.sequence_length = sequence_length
            self.inference = False
            self.time_dist = True

        self.model = self.build()
        self.model.summary()
        self.run_name = self.get_run_name()
        self.reset_lstm_states()

    def build(self):
        """Builds the MDRNN model for training or inference."""
        if self.inference:
            state_input_output = True
        else:
            state_input_output = False
        data_input = tf.keras.layers.Input(
            shape=(self.sequence_length, self.dimension), name="inputs"
        )
        lstm_in = data_input  # starter input for lstm
        state_inputs = []  # storage for LSTM state inputs
        state_outputs = []  # storage for LSTM state outputs

        for layer_i in range(self.n_rnn_layers):
            return_sequences = True
            if (layer_i == self.n_rnn_layers - 1) and not self.time_dist:
                # return sequences false if last layer, and not time distributed.
                return_sequences = False
            state_input = None
            if state_input_output:
                state_h_input = tf.keras.layers.Input(
                    shape=(self.n_hidden_units,), name=f"state_h_{layer_i}"
                )
                state_c_input = tf.keras.layers.Input(
                    shape=(self.n_hidden_units,), name=f"state_c_{layer_i}"
                )
                state_input = [state_h_input, state_c_input]
                state_inputs += state_input
            lstm_out, state_h_output, state_c_output = tf.keras.layers.LSTM(
                self.n_hidden_units,
                name=f"lstm_{layer_i}",
                return_sequences=return_sequences,
                return_state=True,  # state_input_output # better to keep these outputs and just not use.
            )(lstm_in, initial_state=state_input)
            lstm_in = lstm_out
            state_outputs += [state_h_output, state_c_output]

        mdn_layer = mdn.MDN(self.dimension, self.n_mixtures, name="mdn_outputs")
        if self.time_dist:
            mdn_layer = tf.keras.layers.TimeDistributed(mdn_layer, name="td_mdn")
        mdn_out = mdn_layer(lstm_out)  # apply mdn
        if self.inference:
            # for inference, need to track state of the model
            inputs = [data_input] + state_inputs
            outputs = [mdn_out] + state_outputs
        else:
            # for training we don't need to keep track of state in the model
            inputs = data_input
            outputs = mdn_out
        new_model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name=self.model_name()
        )

        if not self.inference:
            # only need loss function and compile when training
            loss_func = mdn.get_mixture_loss_func(self.dimension, self.n_mixtures)
            optimizer = tf.keras.optimizers.Adam()
            new_model.compile(loss=loss_func, optimizer=optimizer)

        return new_model

    def reset_lstm_states(self):
        states = []
        for i in range(self.n_rnn_layers):
            states += [
                np.zeros((1, self.n_hidden_units), dtype=np.float32),
                np.zeros((1, self.n_hidden_units), dtype=np.float32),
            ]
        assert (
            len(states) == self.n_rnn_layers * 2
        ), "length of states list needs to be RNN layers times 2 (h and c for each)"
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

    def load_model(self, model_file=None, model_dir="models"):
        if model_file is None:
            model_file = model_dir + "/" + self.model_name() + ".h5"
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

    def train(
        self,
        X,
        y,
        batch_size=100,
        epochs=10,
        checkpointing=False,
        early_stopping=True,
        save_location="models",
        validation_split=0.1,
        patience=10,
        logging=True,
    ):
        """Train the network for a number of epochs with a specific dataset."""
        # Setup callbacks
        date_string = datetime.datetime.today().strftime("%Y%m%d-%H_%M_%S")
        checkpoint_path = save_location + "/" + self.model_name() + "-ckpt.keras"
        # checkpoint_path = save_location + "/" + model_name + "-E{epoch:02d}-VL{val_loss:.2f}.keras"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )
        terminateOnNaN = tf.keras.callbacks.TerminateOnNaN()
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=patience
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=save_location + "/" + date_string + self.model_name(),
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        )
        callbacks = [terminateOnNaN]
        if checkpointing:
            callbacks.append(checkpoint_callback)
        if early_stopping:
            callbacks.append(early_stopping_callback)
        if logging:
            callbacks.append(tensorboard_callback)

        # Do the data scaling in here.
        X = np.array(X) * SCALE_FACTOR
        y = np.array(y) * SCALE_FACTOR

        ## print out stats.
        print("Number of training examples:")
        print("X:", X.shape)
        print("y:", y.shape)

        # Train
        history = self.model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
        )
        return history

    def generate_touch(self, prev_sample):
        """Generate one forward prediction from a previous sample in format
        (dt, x_1,...,x_n). Pi and Sigma temperature are adjustable."""
        assert (
            len(prev_sample) == self.dimension
        ), "Only works with samples of the same dimension as the network"
        # print("Input sample", prev_sample)
        input_list = [
            prev_sample.reshape(1, 1, self.dimension) * SCALE_FACTOR
        ] + self.lstm_states
        model_output = self.model(input_list)
        mdn_params = model_output[0][0].numpy()
        self.lstm_states = model_output[1:]  # update storage of LSTM state

        # sample from the MDN:
        new_sample = (
            mdn.sample_from_output(
                mdn_params,
                self.dimension,
                self.n_mixtures,
                temp=self.pi_temp,
                sigma_temp=self.sigma_temp,
            )
            / SCALE_FACTOR
        )
        new_sample = new_sample.reshape(
            self.dimension,
        )
        return new_sample
