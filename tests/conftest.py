from impsy import dataset
from impsy import train
from impsy import utils
import os
import random
import pytest


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


## Training configuration


@pytest.fixture(scope="session")
def sequence_length():
    return 3


@pytest.fixture(scope="session")
def batch_size():
    return 3


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


## Generate sample log files, dataset files, and models.


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


@pytest.fixture(scope="session")
def sequence_slices(sequence_length, dimension, batch_size):
    x_t_log = utils.generate_data(
        samples=((sequence_length + 1) * batch_size), dimension=dimension
    )
    slices = train.slice_sequence_examples(x_t_log, sequence_length + 1, step_size=1)
    return slices


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


@pytest.fixture(scope="session")
def tflite_file(trained_model):
    return trained_model['tflite_file']


@pytest.fixture(scope="session")
def weights_file(trained_model):
    return trained_model['weights_file']


@pytest.fixture(scope="session")
def keras_file(trained_model):
    return trained_model['keras_file']

