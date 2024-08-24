from impsy import dataset
from impsy import train
import numpy as np
import os
import tensorflow as tf


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
    assert isinstance(trained_model["history"], tf.keras.callbacks.History)
