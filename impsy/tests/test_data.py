from impsy import dataset
from impsy import train
from impsy import tflite_converter
from impsy import utils
from impsy import mdrnn
import numpy as np
import os
import random
import tensorflow as tf
import pytest
from pathlib import Path


## MDRNN configuration

@pytest.fixture(scope="session")
def dimension():
    return 8

@pytest.fixture(scope="session")
def mdrnn_size():
    return "xs"

@pytest.fixture(scope="session")
def units(mdrnn_size):
    return utils.mdrnn_config(mdrnn_size)['units']

@pytest.fixture(scope="session")
def mixtures(mdrnn_size):
    return utils.mdrnn_config(mdrnn_size)['mixes']

@pytest.fixture(scope="session")
def layers(mdrnn_size):
    return utils.mdrnn_config(mdrnn_size)['layers']

## Locations

@pytest.fixture(scope="session")
def dataset_location(tmp_path_factory):
    location = tmp_path_factory.mktemp("datasets")
    return location

@pytest.fixture(scope="session")
def log_location(tmp_path_factory):
    location = tmp_path_factory.mktemp("logs")
    return location

@pytest.fixture(scope="session")
def models_location(tmp_path_factory):
    location = tmp_path_factory.mktemp("models")
    return location


@pytest.fixture(scope="session")
def log_files(log_location, dimension, number=20, events=49):
    """Creates some test log files for dataset testing."""
    assert dimension > 1, "minimum dimension is 2"
    print(f"dimension: {dimension}")
    test_log_files = []
    for i in range(number):
        test_log_file = log_location / f"2024-06-{i:02d}T12-00-00-{dimension}d-mdrnn.log"
        num_events = random.randint(events, events + 10) # generate a random number of events in a range of 10.
        with open(test_log_file, "w") as file:
            for j in range(num_events):
                nums = [random.random() for _ in range(dimension - 1)]
                num_string = ",".join(map(str, nums))
                test_line = f"2024-06-01T12:00:{j:02d},interface,{num_string}\n"
                file.write(test_line)
        test_log_files.append(test_log_file)
    return test_log_files


@pytest.fixture(scope="session")
def dataset_file(log_location, dataset_location, dimension, log_files):
    dataset_filename = dataset.generate_dataset(
        dimension=dimension, source=log_location, destination=dataset_location
    )
    dataset_full_path = dataset_location / dataset_filename
    return dataset_full_path


def test_log_to_examples(dimension, log_files):
    """Tests transform_log_to_sequence_example with a single example"""
    log = dataset.transform_log_to_sequence_example(log_files[0], dimension)
    assert isinstance(log, np.ndarray)
    assert len(log[0]) == dimension


def test_dataset_command(dataset_location, dataset_file):
    """Test the dataset command runs"""
    with np.load(dataset_file, allow_pickle=True) as loaded:
        corpus = loaded["perfs"]
    print("Loaded performances:", len(corpus))
    print("Num touches:", np.sum([len(l) for l in corpus]))


@pytest.fixture(scope="session")
def trained_model(dimension, dataset_file, dataset_location, models_location, mdrnn_size):
    assert os.path.isfile(dataset_file)
    batch_size = 1
    epochs = 1

    # Train using that dataset
    train_output = train.train_mdrnn(
        dimension=dimension,
        dataset_location=dataset_location,
        model_size=mdrnn_size,
        early_stopping=False,
        patience=10,
        num_epochs=epochs,
        batch_size=batch_size,
        save_location=models_location,
        save_model=True,
        save_weights=True,
        save_tflite=True,
    )
    return train_output


def test_train_command(trained_model):
    """Test that a trained model can be constructed."""
    assert os.path.isfile(trained_model["weights_file"])
    assert os.path.isfile(trained_model["keras_file"])
    assert os.path.isfile(trained_model["tflite_file"])
    assert isinstance(trained_model["history"], tf.keras.callbacks.History)


### tflite conversion tests


def test_config_to_tflite(models_location):
    test_config = "configs/default.toml"
    tflite_file = tflite_converter.config_to_tflite(test_config, save_path=models_location)
    assert os.path.exists(tflite_file)


def test_weights_to_model_file(trained_model, dimension, mdrnn_size):
    weights_file = trained_model["weights_file"]
    print(f"Weights file: {weights_file}")
    tflite_file = tflite_converter.weights_file_to_model_file(weights_file, mdrnn_size, dimension)
    print(f"File returned: {tflite_file}")
    assert os.path.exists(tflite_file)


def test_model_file_to_tflite(trained_model):
    model_filename = trained_model["keras_file"]
    tflite_file = tflite_converter.model_file_to_tflite(model_filename)
    assert os.path.exists(tflite_file)


### inference model tests.


@pytest.fixture(scope="session")
def tflite_file(trained_model):
    return trained_model['tflite_file']

@pytest.fixture(scope="session")
def weights_file(trained_model):
    return trained_model['weights_file']

@pytest.fixture(scope="session")
def keras_file(trained_model):
    return trained_model['keras_file']

@pytest.fixture(scope="session")
def tflite_model(tflite_file, dimension, units, mixtures, layers):
    model = mdrnn.TfliteMDRNN(tflite_file, dimension, units, mixtures, layers)
    return model

def test_tflite_predictions(tflite_model: mdrnn.TfliteMDRNN):
    """Test inference from a TfliteMDRNN model"""
    num_test_steps = 5
    dimension = tflite_model.dimension
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = tflite_model.generate(value)
        assert len(value) == dimension
        value = mdrnn.proc_generated_touch(value, dimension)
        assert len(value) == dimension

@pytest.fixture(scope="session")
def keras_model(keras_file, dimension, units, mixtures, layers):
    model = mdrnn.KerasMDRNN(keras_file, dimension, units, mixtures, layers)
    return model

def test_keras_predictions(keras_model: mdrnn.KerasMDRNN):
    """Test inference from a KerasMDRNN model"""
    num_test_steps = 5
    dimension = keras_model.dimension
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = keras_model.generate(value)
        assert len(value) == dimension
        value = mdrnn.proc_generated_touch(value, dimension)
        assert len(value) == dimension

@pytest.fixture(scope="session")
def weights_model(weights_file, dimension, units, mixtures, layers):
    assert weights_file.suffix == ".h5", "has to be an .h5 weights"
    model = mdrnn.KerasMDRNN(weights_file, dimension, units, mixtures, layers)
    return model

def test_weights_predictions(weights_model: mdrnn.KerasMDRNN):
    """Test inference from a KerasMDRNN model"""
    num_test_steps = 5
    dimension = weights_model.dimension
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = weights_model.generate(value)
        assert len(value) == dimension
        value = mdrnn.proc_generated_touch(value, dimension)
        assert len(value) == dimension


def test_dummy_model():
    model = mdrnn.DummyMDRNN(Path("/"), 4, 64, 5, 2)
    dimension = model.dimension
    value = mdrnn.random_sample(out_dim=dimension)
    value = model.generate(value)
    assert len(value) == dimension

