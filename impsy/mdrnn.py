"""
IMPSY MDRNN Model.
Charles P. Martin, 2018
University of Oslo, Norway.
"""

import numpy as np
import tensorflow as tf
import keras_mdn_layer as mdn
import datetime
from pathlib import Path
import abc
import click


NET_MODE_TRAIN = "train"
NET_MODE_RUN = "run"
LOG_PATH = "./logs/"
SCALE_FACTOR = 10  # scales input and output from the model. Should be the same between training and inference.


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


def lstm_blank_states(layers: int, units: int):
    """Create blank LSTM states for a networks with a number of layers and the same number of LSTM units in each layer"""
    states = []
    for i in range(layers):
        states += [
            np.zeros((1, units), dtype=np.float32),
            np.zeros((1, units), dtype=np.float32),
        ]
    assert (
        len(states) == layers * 2
    ), "length of states list needs to be RNN layers times 2 (h and c for each)"
    return states


def mdrnn_model_name(dimension: int, n_rnn_layers: int, n_hidden_units: int, n_mixtures: int) -> str:
    """Returns the name of a model using it's parameters"""
    name = f"musicMDRNN"
    name += f"-dim{dimension}"
    name += f"-layers{n_rnn_layers}"
    name += f"-units{n_hidden_units}"
    name += f"-mixtures{n_mixtures}"
    name += f"-scale{SCALE_FACTOR}"
    return name


def build_mdrnn_model(dimension: int, n_hidden_units: int, n_mixtures: int, n_layers: int, inference: bool, seq_length = 30):
    """Builds a Keras MDRNN model with specified parameters.
    Can either be a training model or inference model which affects the configured 
    sequence length and whether a loss function is added.
    """
    # Set parameters for inference/training versions.
    if inference:
        state_input_output = True
        sequence_length = 1
        time_dist = False
    else:
        state_input_output = False
        sequence_length = seq_length
        time_dist = True
    # inputs
    data_input = tf.keras.layers.Input(
        shape=(sequence_length, dimension), name="inputs"
    )
    lstm_in = data_input  # starter input for lstm
    state_inputs = []  # storage for LSTM state inputs
    state_outputs = []  # storage for LSTM state outputs

    # lstm layers
    for layer_i in range(n_layers):
        return_sequences = True
        if (layer_i == n_layers - 1) and not time_dist:
            # return sequences false if last layer, and not time distributed.
            return_sequences = False
        state_input = None
        if state_input_output:
            state_h_input = tf.keras.layers.Input(
                shape=(n_hidden_units,), name=f"state_h_{layer_i}"
            )
            state_c_input = tf.keras.layers.Input(
                shape=(n_hidden_units,), name=f"state_c_{layer_i}"
            )
            state_input = [state_h_input, state_c_input]
            state_inputs += state_input
        lstm_out, state_h_output, state_c_output = tf.keras.layers.LSTM(
            n_hidden_units,
            name=f"lstm_{layer_i}",
            return_sequences=return_sequences,
            return_state=True,  # state_input_output # better to keep these outputs and just not use.
        )(lstm_in, initial_state=state_input)
        lstm_in = lstm_out
        state_outputs += [state_h_output, state_c_output]

    # mdn layer
    mdn_layer = mdn.MDN(dimension, n_mixtures, name="mdn_outputs")
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
    name = mdrnn_model_name(dimension, n_layers, n_hidden_units, n_mixtures)
    new_model = tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name=name
    )

    if not inference:
        # only need loss function and compile when training
        loss_func = mdn.get_mixture_loss_func(dimension, n_mixtures)
        optimizer = tf.keras.optimizers.Adam()
        new_model.compile(loss=loss_func, optimizer=optimizer)

    return new_model


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
        """
        # network parameters
        self.dimension = dimension
        self.mode = mode
        self.n_hidden_units = n_hidden_units
        self.n_rnn_layers = layers
        self.n_mixtures = n_mixtures  # number of mixtures
        self.sequence_length = sequence_length # only needed for training.

        # Sampling hyperparameters
        self.pi_temp = 1.5
        self.sigma_temp = 0.01

        # setup model
        if self.mode == NET_MODE_RUN:
            self.inference = True
            self.training = False
        elif self.mode == NET_MODE_TRAIN:
            self.inference = False
            self.training = True
        
        self.model = build_mdrnn_model(self.dimension, self.n_hidden_units, self.n_mixtures, self.n_rnn_layers, self.inference, self.sequence_length)
        self.model_name = mdrnn_model_name(self.dimension, self.n_rnn_layers, self.n_hidden_units, self.n_mixtures)
        self.model.summary()
        self.reset_lstm_states()


    def reset_lstm_states(self):
        self.lstm_states = lstm_blank_states(self.n_rnn_layers, self.n_hidden_units)


    def load_model(self, model_file):
        try:
            self.model.load_weights(model_file)
        except OSError as err:
            click.secho(f"Error loading MDRNN from file: {err}", fg="red")
            click.secho(f"Using untrained MDRNN", fg="red")


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
        save_location = Path(save_location)
        checkpoint_path = save_location / f"{self.model_name}-ckpt.keras"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
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
            log_dir=save_location / f"{date_string}{self.model_name}",
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

    def generate(self, prev_sample):
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
        # Note that we have confirmed that model.__call__() is way faster than model.predict().
        # model_output = self.model.predict(input_list)
        mdn_params = model_output[0][0].numpy()
        # mdn_params = model_output[0][0]
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


class MDRNNInferenceModel(abc.ABC):
    """Abstract class for IMPSY inferences models."""

    model_file: Path
    dimension: int
    n_hidden_units: int
    n_mixtures: int
    n_layers: int

    def __init__(
        self,
        file: Path,
        dimension: int,
        n_hidden_units: int,
        n_mixtures: int,
        n_layers: int,
    ) -> None:
        self.model_file = file
        self.dimension = dimension
        self.n_hidden_units = n_hidden_units
        self.n_mixtures = n_mixtures
        self.n_layers = n_layers
        self.model_name = mdrnn_model_name(self.dimension, self.n_layers, self.n_hidden_units, self.n_mixtures)
        # sampling hyperparameters
        self.pi_temp = 1.5
        self.sigma_temp = 0.01
        self.reset_lstm_states()
        self.prepare() # load the network files.


    def reset_lstm_states(self):
        self.lstm_states = lstm_blank_states(self.n_layers, self.n_hidden_units)
    

    @abc.abstractmethod
    def prepare(self) -> None:
        """Prepare for making predictions."""
        pass


    @abc.abstractmethod
    def generate(self, prev_value: np.ndarray) -> np.ndarray:
        """Handles input values (synchronously) if needed."""
        pass


class TfliteMDRNN(MDRNNInferenceModel):
    """Loads an MDRNN from a tensorflow lite (.tflite) file for running predictions efficiently."""


    def __init__(self, file: Path, dimension: int, n_hidden_units: int, n_mixtures: int, n_layers: int) -> None:
        super().__init__(file, dimension, n_hidden_units, n_mixtures, n_layers)
    

    def prepare(self) -> None:
        assert self.model_file.suffix == ".tflite", "TfliteMDRNN only works on .tflite files."
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_file))
        self.signatures = self.interpreter.get_signature_list()
        self.runner = self.interpreter.get_signature_runner()


    def generate(self, prev_value: np.ndarray) -> np.ndarray:
        """makes a prediction. Needs to know the exact state names at the moment."""
        input_value = prev_value.reshape(1,1,self.dimension) * SCALE_FACTOR
        input_value = input_value.astype(np.float32, copy=False)
        ## Create the input dictionary:
        runner_input = {'inputs': input_value}
        for i in range(self.n_layers):
            runner_input[f'state_h_{i}'] = self.lstm_states[2 * i] # h
            runner_input[f'state_c_{i}'] = self.lstm_states[2 * i + 1] # c
        ## Run inference
        raw_out = self.runner(**runner_input)
        ## Extract the lstm states and mdn parameters
        for i in range(self.n_layers):
            self.lstm_states[2 * i] = raw_out[f'lstm_{i}'] # h
            self.lstm_states[2 * i + 1] = raw_out[f'lstm_{i}_1'] # c
        mdn_params = raw_out['mdn_outputs'].squeeze()
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


class  KerasMDRNN(MDRNNInferenceModel):
    """Loads an MDRNN in inference mode from a .keras file."""


    def __init__(self, file: Path, dimension: int, n_hidden_units: int, n_mixtures: int, n_layers: int) -> None:
        super().__init__(file, dimension, n_hidden_units, n_mixtures, n_layers)


    def prepare(self) -> None:
        assert self.model_file.suffix == ".keras" or self.model_file.suffix == ".h5", "KerasMDRNN only works on .keras or .h5 files."
        if self.model_file.suffix == ".keras":
            # Loading model for .keras files
            self.model = tf.keras.saving.load_model(
                str(self.model_file), 
                custom_objects={"MDN": mdn.MDN}
            )
        elif self.model_file.suffix == ".h5":
            # Loading model for .h5 files
            self.model = build_mdrnn_model(self.dimension, self.n_hidden_units, self.n_mixtures, self.n_layers, inference=True, seq_length=1)
            self.model.load_weights(self.model_file)


    def generate(self, prev_value: np.ndarray) -> np.ndarray:
        """Generate one forward prediction from a previous sample in format
        (dt, x_1,...,x_n). Pi and Sigma temperature are adjustable."""
        assert (
            len(prev_value) == self.dimension
        ), "Only works with samples of the same dimension as the network"
        # print("Input sample", prev_value)
        input_list = [
            prev_value.reshape(1, 1, self.dimension) * SCALE_FACTOR
        ] + self.lstm_states
        model_output = self.model(input_list)
        # Note that we have confirmed that model.__call__() is way faster than model.predict().
        # model_output = self.model.predict(input_list)
        mdn_params = model_output[0][0].numpy()
        # mdn_params = model_output[0][0]
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


class DummyMDRNN(MDRNNInferenceModel):
    """A dummy MDRNN for use if there is no model available (yet or ever). It just generates the same value over and over again."""


    def __init__(self, file: Path, dimension: int, n_hidden_units: int, n_mixtures: int, n_layers: int) -> None:
        super().__init__(file, dimension, n_hidden_units, n_mixtures, n_layers)


    def prepare(self) -> None:
        self.output_value = random_sample(out_dim=self.dimension)


    def generate(self, prev_value: np.ndarray) -> np.ndarray:
        return self.output_value
    
