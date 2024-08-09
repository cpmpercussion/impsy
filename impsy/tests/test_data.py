from impsy import dataset
from impsy import train
from impsy import tflite_converter
import numpy as np
import os
import random
import tensorflow as tf
import pytest


@pytest.fixture
def dimension():
    return 8

@pytest.fixture
def mdrnn_size():
    return "xs"

@pytest.fixture
def test_dir():
    return "tests"

@pytest.fixture
def dataset_location(test_dir):
    location = os.path.join(test_dir, "datasets")
    os.makedirs(location, exist_ok=True)
    return location

@pytest.fixture
def log_location(test_dir):
    location = os.path.join(test_dir, "logs")
    os.makedirs(location, exist_ok=True)
    return location

@pytest.fixture
def models_location(test_dir):
    location = os.path.join(test_dir, "models")
    os.makedirs(location, exist_ok=True)
    return location


@pytest.fixture
def log_files(log_location, dimension, number=20, events=55):
    """Creates some test log files for dataset testing."""
    assert dimension > 1, "minimum dimension is 2"
    print(f"dimension: {dimension}")
    test_log_files = []
    for i in range(number):
        test_log_file = os.path.join(log_location, f"2024-06-{i:02d}T12-00-00-{dimension}d-mdrnn.log")
        with open(test_log_file, "w") as file:
            for j in range(events):
                nums = [random.random() for _ in range(dimension - 1)]
                num_string = ",".join(map(str, nums))
                test_line = f"2024-06-01T12:00:{j:02d},interface,{num_string}\n"
                file.write(test_line)
        test_log_files.append(test_log_file)
    yield test_log_files
    # after test, delete the files.
    for f in test_log_files:
        os.remove(f)


@pytest.fixture
def dataset_file(log_location, dataset_location, dimension, log_files):
    dataset_filename = dataset.generate_dataset(
        dimension=dimension, source=log_location, destination=dataset_location
    )
    dataset_full_path = os.path.join(dataset_location, dataset_filename)
    yield dataset_full_path
    # finally, remove the dataset
    os.remove(dataset_full_path)


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


@pytest.fixture
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
    )

    assert isinstance(train_output["history"], tf.keras.callbacks.History)
    assert os.path.isfile(train_output["weights_file"])
    assert os.path.isfile(train_output["keras_file"])

    yield train_output

    # clean up
    os.remove(train_output["weights_file"])
    os.remove(train_output["keras_file"])


def test_train_command(trained_model):
    """Test the training command"""
    # just assert that the model output exists.

    assert os.path.isfile(trained_model["weights_file"])
    assert os.path.isfile(trained_model["keras_file"])
    assert isinstance(trained_model["history"], tf.keras.callbacks.History)


### tflite tests

def test_config_to_tflite():
    test_config = "configs/default.toml"
    expected_output_path = (
        "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
    )
    tflite_converter.config_to_tflite(test_config)
    assert os.path.exists(expected_output_path)
    os.remove(expected_output_path)


def test_weights_to_model_file(trained_model, dimension, test_dir, mdrnn_size):
    weights_file = trained_model["weights_file"]
    print(f"Weights file: {weights_file}")
    model_file_name = tflite_converter.weights_file_to_model_file(weights_file, mdrnn_size, dimension, test_dir)
    print(f"File returned: {model_file_name}")
    assert os.path.exists(model_file_name)
    os.remove(model_file_name)


def test_model_file_to_tflite(trained_model):
    model_filename = trained_model["keras_file"]
    expected_output_path = model_filename.removesuffix('.keras') + '.tflite'
    tflite_converter.model_file_to_tflite(model_filename)
    assert os.path.exists(expected_output_path)
    os.remove(expected_output_path)
