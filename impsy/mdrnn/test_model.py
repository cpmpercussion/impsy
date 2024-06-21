from . import PredictiveMusicMDRNN, NET_MODE_RUN, NET_MODE_TRAIN
from . import slice_sequence_examples, seq_to_overlapping_format, random_sample
from . import sample_data
import tensorflow as tf
import numpy as np


def test_model():
    """Test creation of a PredictiveMusicMDRNN Model."""
    net = PredictiveMusicMDRNN()
    assert isinstance(net, PredictiveMusicMDRNN)


def test_inference():
    """Test inference from a PredictiveMusicMDRNN model"""
    dimension = 8
    num_test_steps = 20
    net = PredictiveMusicMDRNN(mode=NET_MODE_RUN, dimension=dimension)
    value = random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = net.generate_touch(value)
    assert len(value) == dimension


def test_training():
    """Test training on a PredictiveMusicMDRNN model"""
    num_epochs = 2
    sequence_length = 100
    net = PredictiveMusicMDRNN(
        mode=NET_MODE_TRAIN,
        dimension=2,
        n_hidden_units=128,
        n_mixtures=5,
        batch_size=100,
        sequence_length=sequence_length,
        layers=2,
    )
    x_t_log = sample_data.generate_data(samples=((sequence_length + 1) * 10))
    slices = slice_sequence_examples(x_t_log, sequence_length + 1, step_size=1)
    Xs, ys = seq_to_overlapping_format(slices)
    history = net.train(Xs, ys, num_epochs=num_epochs, saving=False)
    assert isinstance(history, tf.keras.callbacks.History)
