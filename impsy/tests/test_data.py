from impsy import dataset
from impsy import train
import numpy as np
import os
import random
import tensorflow as tf


def create_test_log_files(location="tests/logs", dimension=4, number=10, events=50):
    """Creates some test log files for dataset testing."""
    assert dimension > 1, "minimum dimension is 2"
    os.makedirs(location, exist_ok=True)
    log_files = []
    for i in range(number):
        test_log_file = f"{location}/2024-06-{i:02d}T12-00-00-4d-mdrnn.log"
        with open(test_log_file, "w") as file:
            for j in range(events):
                nums = [random.random() for _ in range(dimension - 1)]
                num_string = ",".join(map(str, nums))
                test_line = f"2024-06-01T12:00:{j:02d},interface,{num_string}\n"
                file.write(test_line)
        log_files.append(test_log_file)
    return log_files


def remove_test_log_files(log_files):
    """Deletes test log files"""
    for f in log_files:
        os.remove(f)


def test_log_to_examples():
    """Tests transform_log_to_sequence_example with a single example"""
    dimension = 8
    log_files = create_test_log_files(number=1, dimension=dimension)
    log = dataset.transform_log_to_sequence_example(log_files[0], dimension)
    remove_test_log_files(log_files)
    assert isinstance(log, np.ndarray)
    assert len(log[0]) == dimension


def test_dataset_command():
    """Test the dataset command runs"""
    test_log_area = "tests/logs"
    test_dataset_area = "tests/datasets"
    os.makedirs(test_dataset_area, exist_ok=True)
    dimension = 4
    log_files = create_test_log_files(
        location=test_log_area, dimension=dimension, number=10
    )
    dataset.generate_dataset(
        dimension=dimension, source="tests/logs", destination=test_dataset_area
    )
    remove_test_log_files(log_files)


def test_train_command():
    """Test the training command"""

    dimension = 4
    number_files = 20
    model_size = "xs"
    batch_size = 1
    epochs = 1

    # Setup directories
    log_area = "tests/logs"
    dataset_area = "tests/datasets"
    model_area = "tests/models"
    for d in [log_area, dataset_area, model_area]:
        os.makedirs(d, exist_ok=True)

    # Generate Log Files
    log_files = create_test_log_files(
        location=log_area, dimension=dimension, number=number_files, events=55
    )

    # Generate a dataset
    dataset_file = dataset.generate_dataset(
        dimension=dimension, source=log_area, destination=dataset_area
    )

    # Clean up log files
    remove_test_log_files(log_files)

    # Train using that dataset
    history = train.train_mdrnn(
        dimension=dimension,
        dataset_location=dataset_area,
        model_size=model_size,
        early_stopping=False,
        patience=10,
        num_epochs=epochs,
        batch_size=batch_size,
        save_location=model_area,
    )

    os.remove(dataset_area + "/" + dataset_file)

    assert isinstance(history, tf.keras.callbacks.History)
