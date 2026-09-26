"""impsy.train: Functions for training an impsy mdrnn model."""

from .utils import mdrnn_config
import numpy as np
import random
import click
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


def dataset_dimension(dataset_file) -> int:
    """Returns the dimension (dt plus number of values) of the data in a .npz dataset."""
    with np.load(dataset_file, allow_pickle=True) as loaded:
        corpus = loaded["perfs"]
    for perf in corpus:
        if len(perf) > 0:
            return int(np.asarray(perf).shape[1])
    raise ValueError(f"Dataset {dataset_file} has no data.")


def find_dataset_file(dataset_location, dimension: int | None = None) -> Path:
    """Resolves a dataset file from a .npz path or a directory of datasets.

    For a directory, uses training-dataset-{dimension}d.npz if dimension is given,
    otherwise the only .npz file in the directory.
    """
    dataset_location = Path(dataset_location)
    if dataset_location.suffix == ".npz":
        return dataset_location
    if dimension is not None:
        return dataset_location / f"training-dataset-{dimension}d.npz"
    candidates = sorted(dataset_location.glob("*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Found {len(candidates)} .npz files in {dataset_location}; "
        "give a dataset file or a dimension to choose one."
    )


def train_mdrnn(
    dimension: int | None,
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
    callbacks: list | None = None,
):
    """Loads a dataset, creates a model and runs the training procedure.

    dimension can be None to read it from the dataset. If training is interrupted
    (e.g., with Ctrl-C) the model is still saved with the latest weights.
    Extra Keras callbacks can be given to follow or stop training.
    """
    from . import mdrnn

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
    dataset_location = find_dataset_file(dataset_location, dimension)
    click.secho(f"Dataset: {dataset_location}")
    data_dimension = dataset_dimension(dataset_location)
    if dimension is None:
        dimension = data_dimension
    elif dimension != data_dimension:
        raise ValueError(
            f"Dataset has dimension {data_dimension} but dimension {dimension} was requested."
        )
    click.secho(f"Dimension: {dimension}", fg="blue")
    with np.load(dataset_location, allow_pickle=True) as loaded:
        corpus = loaded["perfs"]
    print("Loaded performances:", len(corpus))
    print("Num touches:", np.sum([len(l) for l in corpus]))

    # Restrict corpus to performances longer than the training sequence length.
    corpus = [l for l in corpus if len(l) > SEQ_LEN + 1]
    click.secho(f"Corpus Examples: {len(corpus)}", fg="blue")
    if not corpus:
        raise ValueError(
            f"No performances in the dataset are longer than {SEQ_LEN + 1} events, "
            "so there's nothing to train on. Record some longer logs."
        )

    # Prepare training data as X and Y.
    slices = []
    for seq in corpus:
        slices += slice_sequence_examples(seq, SEQ_LEN + 1, step_size=SEQ_STEP)
    X, y = seq_to_overlapping_format(slices)

    # Setup Training Model
    training_mdrnn = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_TRAIN,
        dimension=dimension,
        n_hidden_units=mdrnn_units,
        n_mixtures=mdrnn_mixes,
        sequence_length=SEQ_LEN,
        layers=mdrnn_layers,
    )

    # Setup Inference Model
    inference_mdrnn = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_RUN,
        dimension=dimension,
        n_hidden_units=mdrnn_units,
        n_mixtures=mdrnn_mixes,
        sequence_length=1,
        layers=mdrnn_layers,
    )

    validation_split = 0.10
    try:
        history = training_mdrnn.train(
            X,
            y,
            batch_size=batch_size,
            epochs=num_epochs,
            checkpointing=True,
            early_stopping=early_stopping,
            save_location=save_location,
            validation_split=validation_split,
            patience=patience,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        click.secho("Training interrupted, saving the latest weights.", fg="yellow")
        history = getattr(training_mdrnn.model, "history", None)

    # Save final Model
    model_name = training_mdrnn.model_name

    # start preparing output dict output in case
    output = {
        "name": model_name,
        "history": history,
    }

    # Don't save h5 weights anymore, only using .keras and .tflite files.
    if save_weights:
        # Save .h5 file
        model_weights_file = save_location / f"{model_name}.weights.h5"
        training_mdrnn.model.save_weights(model_weights_file)
        output["weights_file"] = model_weights_file

    if save_model:
        # Save .keras file
        trained_weights = training_mdrnn.model.get_weights()
        model_name = inference_mdrnn.model_name
        model_keras_file = save_location / f"{model_name}.keras"
        inference_mdrnn.model.set_weights(trained_weights)
        inference_mdrnn.model.save(model_keras_file)
        output["keras_file"] = model_keras_file

    if save_tflite:
        # Save .tflite file
        from .tflite_converter import model_to_tflite

        if not save_model:
            inference_mdrnn.model.set_weights(training_mdrnn.model.get_weights())
        tflite_path = save_location / f"{inference_mdrnn.model_name}.tflite"
        tflite_file = model_to_tflite(inference_mdrnn.model, tflite_path)
        output["tflite_file"] = tflite_file

    return output


@click.command(name="train")
@click.argument("dataset", required=False, default=None)
@click.option(
    "-D",
    "--dimension",
    type=int,
    default=None,
    help="The dimension of the data to model, must be >= 2 (read from the dataset if not given).",
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
@click.option(
    "-O",
    "--destination",
    type=str,
    default="models",
    help="The destination directory to write trained model files to.",
)
def train(
    dataset: str | None,
    dimension: int | None,
    source: str,
    modelsize: str,
    earlystopping: bool,
    patience: int,
    numepochs: int,
    batchsize: int,
    destination: str,
):
    """Trains an IMPSY MDRNN model based on an existing dataset (run dataset command first!).

    DATASET is an optional .npz file to train on (overrides --source). The trained
    model is saved in .keras and .tflite formats. Press Ctrl-C to stop training
    early and save the model as it is.
    """
    if dataset is not None:
        source = dataset
    click.secho(
        f"IMPSY: Going to train a {modelsize} sized MDRNN model.",
        fg="green",
    )
    Path(destination).mkdir(parents=True, exist_ok=True)
    output = train_mdrnn(
        dimension,
        source,
        modelsize,
        earlystopping,
        patience,
        numepochs,
        batchsize,
        save_location=destination,
    )
    click.secho("IMPSY: training completed.", fg="green")
    click.secho(f"Model file: {output['tflite_file']}", fg="green")
