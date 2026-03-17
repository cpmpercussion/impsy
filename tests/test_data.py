import tensorflow as tf
import numpy as np
import os
from impsy import dataset
from impsy import train


## Test logs, data and training.


def test_data_munging(sequence_length, batch_size, dimension, sequence_slices):
    """Test the data munging functions"""

    # overlapping
    Xs, ys = train.seq_to_overlapping_format(sequence_slices)
    assert len(Xs) == len(ys)
    assert len(Xs[0]) == sequence_length
    assert len(ys[0]) == sequence_length

    print("Xs:", len(Xs[0]))
    print("ys:", len(ys[0]))

    # singleton
    X, y = train.seq_to_singleton_format(sequence_slices)
    print("X:", len(X[0]))
    print("y:", len(y[0]))

    assert len(X) == len(y)
    assert len(X[0]) == sequence_length
    assert len(y[0]) == dimension


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


def test_train_command(trained_model):
    """Test that a trained model can be constructed."""
    assert os.path.isfile(trained_model["weights_file"])
    assert os.path.isfile(trained_model["keras_file"])
    assert os.path.isfile(trained_model["tflite_file"])


def test_malformed_log_file(tmp_path, dimension):
    """Test that a malformed log file is handled gracefully during dataset generation."""
    # Create a malformed log file
    log_file = tmp_path / f"2024-01-01T12-00-00-{dimension}d-mdrnn.log"
    log_file.write_text("this is not,valid,csv data\nbad,data,here\n")
    # Should not crash
    result = dataset.generate_dataset(dimension=dimension, source=str(tmp_path), destination=str(tmp_path))
    # Result should be None since no valid data was produced
    assert result is None


def test_empty_log_directory(tmp_path, dimension):
    """Test dataset generation with no matching log files."""
    result = dataset.generate_dataset(dimension=dimension, source=str(tmp_path), destination=str(tmp_path))
    assert result is None


def test_log_with_rnn_source_filtered(tmp_path):
    """Test that 'rnn' source lines are filtered out of log data."""
    dimension = 3
    log_file = tmp_path / f"2024-01-01T12-00-00-{dimension}d-mdrnn.log"
    lines = []
    for i in range(10):
        lines.append(f"2024-01-01T12:00:{i:02d},interface,{0.5},{0.5}\n")
        lines.append(f"2024-01-01T12:00:{i:02d},rnn,{0.9},{0.9}\n")  # should be filtered
    log_file.write_text("".join(lines))
    log = dataset.transform_log_to_sequence_example(str(log_file), dimension)
    # All entries should be from interface source only
    assert log.shape[0] == 9  # 10 interface lines minus 1 (first diff is NaN)
