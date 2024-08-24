from impsy import mdrnn
from impsy import train
from impsy import utils
import tensorflow as tf
import pytest
from pathlib import Path


## PredictiveMusicMDRNN testing.


def test_inference():
    """Test inference from a PredictiveMusicMDRNN model"""
    dimension = 8
    num_test_steps = 5
    net = mdrnn.PredictiveMusicMDRNN(mode=mdrnn.NET_MODE_RUN, dimension=dimension)
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = net.generate(value)
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


def test_model_config():
    """Tests the model config function."""
    conf = utils.mdrnn_config('s')
    assert conf["units"] == utils.SIZE_TO_PARAMETERS['s']['units']


### inference model tests.


@pytest.fixture(scope="session")
def tflite_model(tflite_file, dimension, units, mixtures, layers):
    model = mdrnn.TfliteMDRNN(tflite_file, dimension, units, mixtures, layers)
    return model

def test_tflite_predictions(tflite_model: mdrnn.TfliteMDRNN):
    """Test inference from a TfliteMDRNN model"""
    num_test_steps = 5
    dimension = tflite_model.dimension
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = tflite_model.generate(value)
        assert len(value) == dimension
        value = mdrnn.proc_generated_touch(value, dimension)
        assert len(value) == dimension

@pytest.fixture(scope="session")
def keras_model(keras_file, dimension, units, mixtures, layers):
    model = mdrnn.KerasMDRNN(keras_file, dimension, units, mixtures, layers)
    return model

def test_keras_predictions(keras_model: mdrnn.KerasMDRNN):
    """Test inference from a KerasMDRNN model"""
    num_test_steps = 5
    dimension = keras_model.dimension
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = keras_model.generate(value)
        assert len(value) == dimension
        value = mdrnn.proc_generated_touch(value, dimension)
        assert len(value) == dimension

@pytest.fixture(scope="session")
def weights_model(weights_file, dimension, units, mixtures, layers):
    assert weights_file.suffix == ".h5", "has to be an .h5 weights"
    model = mdrnn.KerasMDRNN(weights_file, dimension, units, mixtures, layers)
    return model

def test_weights_predictions(weights_model: mdrnn.KerasMDRNN):
    """Test inference from a KerasMDRNN model"""
    num_test_steps = 5
    dimension = weights_model.dimension
    value = mdrnn.random_sample(out_dim=dimension)
    for i in range(num_test_steps):
        value = weights_model.generate(value)
        assert len(value) == dimension
        value = mdrnn.proc_generated_touch(value, dimension)
        assert len(value) == dimension


def test_dummy_model():
    model = mdrnn.DummyMDRNN(Path("/"), 4, 64, 5, 2)
    dimension = model.dimension
    value = mdrnn.random_sample(out_dim=dimension)
    value = model.generate(value)
    assert len(value) == dimension
