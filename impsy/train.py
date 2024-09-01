"""impsy.train: Functions for training an impsy mdrnn model."""

import random
import numpy as np
import click
from .utils import mdrnn_config
from pathlib import Path


# Model training hyperparameters

SEQ_LEN = 50
SEQ_STEP = 1
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


def train_mdrnn(
    dimension: int,
    dataset_location: str,
    model_size: str,
    early_stopping: bool,
    patience: int,
    num_epochs: int,
    batch_size: int,
    save_location: str = "models",
    save_model: bool = True,
    save_weights: bool = False,
    save_tflite: bool = True, 
):
    """Loads a dataset, creates a model and runs the training procedure."""
    import impsy.mdrnn as mdrnn
    from tensorflow import keras
    from .tflite_converter import model_to_tflite


    model_config = mdrnn_config(model_size)
    mdrnn_units = model_config["units"]
    mdrnn_layers = model_config["layers"]
    mdrnn_mixes = model_config["mixes"]

    save_location = Path(save_location)

    click.secho(f"Model size: {model_size}", fg="blue")
    click.secho(f"Units: {mdrnn_units}", fg="blue")
    click.secho(f"Layers: {mdrnn_layers}", fg="blue")
    click.secho(f"Mixtures: {mdrnn_mixes}", fg="blue")

    random.seed(SEED)
    np.random.seed(SEED)

    # Load dataset
    dataset_location = Path(dataset_location)
    dataset_default_name = f"training-dataset-{str(dimension)}d.npz"
    if dataset_location.suffix == "":
        dataset_default_name = f"training-dataset-{str(dimension)}d.npz"
        dataset_location = dataset_location / dataset_default_name
    assert dataset_location.suffix == ".npz", "dataset file to load must end with .npz"
    click.secho(f"Dataset: {dataset_location}")
    with np.load(dataset_location, allow_pickle=True) as loaded:
        corpus = loaded["perfs"]
    print("Loaded performances:", len(corpus))
    print("Num touches:", np.sum([len(l) for l in corpus]))

    # Restrict corpus to performances longer than the training sequence length.
    corpus = [l for l in corpus if len(l) > SEQ_LEN + 1]
    click.secho(f"Corpus Examples: {len(corpus)}", fg="blue")

    # Prepare training data as X and Y.
    slices = []
    for seq in corpus:
        slices += slice_sequence_examples(seq, SEQ_LEN + 1, step_size=SEQ_STEP)
    X, y = seq_to_overlapping_format(slices)

    # Setup Training Model
    mdrnn_manager = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_TRAIN,
        dimension=dimension,
        n_hidden_units=mdrnn_units,
        n_mixtures=mdrnn_mixes,
        sequence_length=SEQ_LEN,
        layers=mdrnn_layers,
    )

    validation_split = 0.10
    history = mdrnn_manager.train(
        X,
        y,
        batch_size=batch_size,
        epochs=num_epochs,
        checkpointing=True,
        early_stopping=early_stopping,
        save_location=save_location,
        validation_split=validation_split,
        patience=patience,
    )

    # Save final Model
    model_name = mdrnn_manager.model_name

    # start preparing output dict output in case
    output = {
        "name": model_name,
        "history": history,
    }

    # Don't save h5 weights anymore, only using .keras and .tflite files.
    if save_weights:
        # Save .h5 file
        model_weights_file = save_location / f"{model_name}.h5"
        mdrnn_manager.model.save_weights(model_weights_file)
        output["weights_file"] = model_weights_file
    
    if save_model:
        # Save .keras file
        trained_weights = mdrnn_manager.model.get_weights()
        inference_mdrnn = mdrnn.PredictiveMusicMDRNN(
            mode=mdrnn.NET_MODE_RUN,
            dimension=dimension,
            n_hidden_units=mdrnn_units,
            n_mixtures=mdrnn_mixes,
            sequence_length=1,
            layers=mdrnn_layers,
        )
        model_name = inference_mdrnn.model_name
        model_keras_file = save_location / f"{model_name}.keras"
        inference_mdrnn.model.set_weights(trained_weights)
        inference_mdrnn.model.save(model_keras_file)
        output["keras_file"] = model_keras_file

    if save_tflite:
        # Save .tflite file
        tflite_file = model_to_tflite(inference_mdrnn.model, model_keras_file)
        output["tflite_file"] = tflite_file

    return output


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
    help="A .npz dataset file to use for training, or source directory to obtain .npz dataset files.",
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
    """Trains an IMPSY MDRNN model based on an existing dataset (run dataset command first!)."""
    click.secho(
        f"IMPSY: Going to train a {dimension}D, {modelsize} sized MDRNN model.",
        fg="green",
    )
    train_mdrnn(
        dimension, source, modelsize, earlystopping, patience, numepochs, batchsize
    )
    click.secho("IMPSY: training completed.", fg="green")
