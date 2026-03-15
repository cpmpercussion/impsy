import tensorflow as tf
from impsy import mdrnn
# from impsy import train
# from impsy import utils
# import pytest
# from pathlib import Path


def test_train_model_building():
    # run the tests.
    n_hidden_units = 32
    dimension = 8
    n_mixtures = 5
    n_layers = 2
    # build a training model
    train_model = mdrnn.build_mdrnn_model(dimension, n_hidden_units, n_mixtures, n_layers=n_layers, inference=False, seq_length=50)
    print(f"Parameters: {train_model.count_params()}")
    assert isinstance(train_model, tf.keras.models.Model)

def test_inference_model_building():
    # run the tests.
    n_hidden_units = 32
    dimension = 8
    n_mixtures = 5
    n_layers = 2
    # build an inference model
    inference_model = mdrnn.build_mdrnn_model(dimension, n_hidden_units, n_mixtures, n_layers=n_layers, inference=True, seq_length = 1)
    print(f"Parameters: {inference_model.count_params()}")
    assert isinstance(inference_model, tf.keras.models.Model)
