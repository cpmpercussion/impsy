from impsy import mdrnn
from impsy import train 
from impsy import utils
import tensorflow as tf


def test_model():
    """Test creation of a PredictiveMusicMDRNN Model."""
    net = mdrnn.PredictiveMusicMDRNN()
    assert isinstance(net, mdrnn.PredictiveMusicMDRNN)


def test_inference():
    """Test inference from a PredictiveMusicMDRNN model"""
    dimension = 8
    num_test_steps = 5
    net = mdrnn.PredictiveMusicMDRNN(mode=mdrnn.NET_MODE_RUN, dimension=dimension)
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = net.generate_touch(value)
    assert len(value) == dimension


def test_training():
    """Test training on a PredictiveMusicMDRNN model"""
    num_epochs = 1
    sequence_length = 3
    batch_size = 3
    dimension = 3
    net = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_TRAIN,
        dimension=dimension,
        n_hidden_units=16,
        n_mixtures=5,
        batch_size=batch_size,
        sequence_length=sequence_length,
        layers=2,
    )
    x_t_log = utils.generate_data(samples=((sequence_length + 1) * batch_size), dimension=dimension)
    slices = train.slice_sequence_examples(x_t_log, sequence_length + 1, step_size=1)
    Xs, ys = train.seq_to_overlapping_format(slices)
    history = net.train(Xs, ys, num_epochs=num_epochs, saving=False)
    assert isinstance(history, tf.keras.callbacks.History)
