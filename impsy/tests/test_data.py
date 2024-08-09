from impsy import dataset
from impsy import train
import numpy as np
import os
import random
import tensorflow as tf
import pytest


@pytest.fixture
def dimension():
    return 8

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


def test_train_command(dimension, dataset_file, dataset_location, models_location):
    """Test the training command"""
    assert os.path.isfile(dataset_file)
    
    model_size = "xs"
    batch_size = 1
    epochs = 1

    # Train using that dataset
    history = train.train_mdrnn(
        dimension=dimension,
        dataset_location=dataset_location,
        model_size=model_size,
        early_stopping=False,
        patience=10,
        num_epochs=epochs,
        batch_size=batch_size,
        save_location=models_location,
    )

    assert isinstance(history, tf.keras.callbacks.History)
