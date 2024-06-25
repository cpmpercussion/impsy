"""impsy.train: Functions for training an impsy mdrnn model."""

import random
import numpy as np
import datetime
import click
from .utils import mdrnn_config


# Model training hyperparameters

SEQ_LEN = 50
SEQ_STEP = 1
TIME_DIST = True
VAL_SPLIT = 0.10
SEED = 2345

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


@click.command(name="train")
@click.option(
    "-D",
    "--dimension",
    type=int,
    default=2,
    help="The dimension of the data to model, must be >= 2.",
)
@click.option(
    "-S",
    "--source",
    type=str,
    default="datasets",
    help="The source directory to obtain .npz dataset files.",
)
@click.option(
    "-M",
    "--modelsize",
    default="s",
    help="The model size: xxs, xs, s, m, l, xl.",
    type=str,
)
@click.option(
    "--earlystopping/--no-earlystopping", default=True, help="Use early stopping."
)
@click.option(
    "-P",
    "--patience",
    type=int,
    default=10,
    help="The number of epochs patience for early stopping.",
)
@click.option(
    "-N", "--numepochs", type=int, default=100, help="The maximum number of epochs."
)
@click.option(
    "-B",
    "--batchsize",
    type=int,
    default=64,
    help="Batch size for training, default=64.",
)
def train(
    dimension: int,
    source: str,
    modelsize: str,
    earlystopping: bool,
    patience: int,
    numepochs: int,
    batchsize: int,
):
    """Trains a predictive music interaction model."""
    import impsy.mdrnn as mdrnn
    from tensorflow import keras

    model_config = mdrnn_config(modelsize)
    mdrnn_units = model_config["units"]
    mdrnn_layers = model_config["layers"]
    mdrnn_mixes = model_config["mixes"]

    print("Model size:", modelsize)
    print("Units:", mdrnn_units)
    print("Layers:", mdrnn_layers)
    print("Mixtures:", mdrnn_mixes)

    random.seed(SEED)
    np.random.seed(SEED)

    # Load dataset
    dataset_location = f"{source}/"
    dataset_filename = f"training-dataset-{str(dimension)}d.npz"
    with np.load(dataset_location + dataset_filename, allow_pickle=True) as loaded:
        corpus = loaded["perfs"]

    print("Loaded performances:", len(corpus))
    print("Num touches:", np.sum([len(l) for l in corpus]))
    # Restrict corpus to performances longer than the training sequence length.
    corpus = [l for l in corpus if len(l) > SEQ_LEN + 1]
    print("Corpus Examples:", len(corpus))
    # Prepare training data as X and Y.
    slices = []
    for seq in corpus:
        slices += mdrnn.slice_sequence_examples(seq, SEQ_LEN + 1, step_size=SEQ_STEP)
    X, y = mdrnn.seq_to_overlapping_format(slices)
    X = np.array(X) * mdrnn.SCALE_FACTOR
    y = np.array(y) * mdrnn.SCALE_FACTOR

    print("Number of training examples:")
    print("X:", X.shape)
    print("y:", y.shape)

    # Setup Training Model
    model = mdrnn.build_model(
        seq_len=SEQ_LEN,
        hidden_units=mdrnn_units,
        num_mixtures=mdrnn_mixes,
        layers=mdrnn_layers,
        out_dim=dimension,
        time_dist=TIME_DIST,
        inference=False,
        print_summary=True,
    )

    model_dir = "models/"
    model_name = (
        "musicMDRNN"
        + "-dim"
        + str(dimension)
        + "-layers"
        + str(mdrnn_layers)
        + "-units"
        + str(mdrnn_units)
        + "-mixtures"
        + str(mdrnn_mixes)
        + "-scale"
        + str(mdrnn.SCALE_FACTOR)
    )
    date_string = datetime.datetime.today().strftime("%Y%m%d-%H_%M_%S")

    # filepath = model_dir + model_name + "-E{epoch:02d}-VL{val_loss:.2f}.keras"
    filepath = model_dir + model_name + "-ckpt.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    terminateOnNaN = keras.callbacks.TerminateOnNaN()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=patience
    )
    tboard = keras.callbacks.TensorBoard(
        log_dir="./logs/" + date_string + model_name,
        histogram_freq=0,
        write_graph=True,
        update_freq="epoch",
    )

    callbacks = [checkpoint, terminateOnNaN, tboard]
    if earlystopping:
        print("Enabling Early Stopping.")
        callbacks.append(early_stopping)
    # Train
    model.fit(
        X,
        y,
        batch_size=batchsize,
        epochs=numepochs,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
    )

    # Save final Model
    model.save_weights(model_dir + model_name + ".h5")

    # ## Converting for tensorflow lite.
    # # Convert the model.
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # tflite_model_name = f`{model_dir}{model_name}-lite.h5`
    # with open(tflite_model_name, 'wb') as f:
    #   f.write(tflite_model)

    print("Training done, bye.")
