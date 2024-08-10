from impsy import mdrnn
from impsy import train
from impsy import utils
import tensorflow as tf
import pytest

@pytest.fixture(scope="session")
def dimension():
    return 3

@pytest.fixture(scope="session")
def sequence_length():
    return 3

@pytest.fixture(scope="session")
def batch_size():
    return 3

@pytest.fixture(scope="session")
def sequence_slices(sequence_length, dimension, batch_size):
    x_t_log = utils.generate_data(
        samples=((sequence_length + 1) * batch_size), dimension=dimension
    )
    slices = train.slice_sequence_examples(x_t_log, sequence_length + 1, step_size=1)
    return slices

def test_inference():
    """Test inference from a PredictiveMusicMDRNN model"""
    dimension = 8
    num_test_steps = 5
    net = mdrnn.PredictiveMusicMDRNN(mode=mdrnn.NET_MODE_RUN, dimension=dimension)
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = net.generate_touch(value)
        proc_touch = mdrnn.proc_generated_touch(value, dimension)
    assert len(value) == dimension
    assert len(proc_touch == dimension)


def test_training(sequence_length, batch_size, dimension, sequence_slices):
    """Test training on a PredictiveMusicMDRNN model"""
    num_epochs = 1
    net = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_TRAIN,
        dimension=dimension,
        n_hidden_units=8,
        n_mixtures=3,
        sequence_length=sequence_length,
        layers=1,
    )
    Xs, ys = train.seq_to_overlapping_format(sequence_slices)
    history = net.train(Xs, ys, batch_size=batch_size, epochs=num_epochs, logging=False)
    assert isinstance(history, tf.keras.callbacks.History)


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


def test_model_config():
    """Tests the model config function."""
    conf = utils.mdrnn_config("s")
    assert conf["units"] == 64
