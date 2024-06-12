from . import *
from . import sample_data
import tensorflow.compat.v1 as tf
import numpy as np


def test_model():
    """Test creation of a PredictiveMusicMDRNN Model."""
    net = PredictiveMusicMDRNN()
    assert isinstance(net, PredictiveMusicMDRNN)


def test_inference():
    """Test inference from a PredictiveMusicMDRNN model"""
    dimension = 8
    net = PredictiveMusicMDRNN(mode=NET_MODE_RUN, dimension=dimension)
    input_value = random_sample(out_dim=dimension)
    output_value = net.generate_touch(input_value)
    assert len(output_value) == dimension


def test_training():
    """Test training on a PredictiveMusicMDRNN model"""
    num_epochs = 1
    sequence_length = 100
    net = PredictiveMusicMDRNN(mode=NET_MODE_TRAIN, 
                               dimension=2, 
                               n_hidden_units=128, 
                               n_mixtures=5, 
                               batch_size=100, 
                               sequence_length=sequence_length, 
                               layers=2)
    x_t_log = sample_data.generate_data(samples=((sequence_length+1)*10))
    slices = slice_sequence_examples(x_t_log, sequence_length+1, step_size=1)
    Xs, ys = seq_to_overlapping_format(slices)
    history = net.train(Xs, ys, num_epochs=num_epochs, saving=False)
    assert isinstance(history, tf.keras.callbacks.History)
    
