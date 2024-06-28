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
        proc_touch = mdrnn.proc_generated_touch(value, dimension)
    assert len(value) == dimension
    assert len(proc_touch == dimension)


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
        sequence_length=sequence_length,
        layers=2,
    )
    x_t_log = utils.generate_data(
        samples=((sequence_length + 1) * batch_size), dimension=dimension
    )
    slices = train.slice_sequence_examples(x_t_log, sequence_length + 1, step_size=1)
    Xs, ys = train.seq_to_overlapping_format(slices)
    history = net.train(Xs, ys, batch_size=batch_size, epochs=num_epochs, logging=False)
    assert isinstance(history, tf.keras.callbacks.History)


def test_data_munging():
    """Test the data munging functions"""
    sequence_length = 50
    batch_size = 100
    dimension = 12

    # get some data
    x_t_log = utils.generate_data(
        samples=((sequence_length + 1) * batch_size * 10), dimension=dimension
    )

    # slice
    slices = train.slice_sequence_examples(x_t_log, sequence_length + 1, step_size=1)

    # overlapping
    Xs, ys = train.seq_to_overlapping_format(slices)
    assert len(Xs) == len(ys)
    assert len(Xs[0]) == sequence_length
    assert len(ys[0]) == sequence_length

    print("Xs:", len(Xs[0]))
    print("ys:", len(ys[0]))

    # singleton
    X, y = train.seq_to_singleton_format(slices)
    print("X:", len(X[0]))
    print("y:", len(y[0]))

    assert len(X) == len(y)
    assert len(X[0]) == sequence_length
    assert len(y[0]) == dimension


def test_model_config():
    """Tests the model config function."""
    conf = utils.mdrnn_config("s")
    assert conf["units"] == 64
