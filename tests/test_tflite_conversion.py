from impsy import tflite_converter
import os


### tflite conversion tests


def test_config_to_tflite(models_location):
    test_config = "configs/default.toml"
    tflite_file = tflite_converter.config_to_tflite(test_config, save_path=models_location)
    assert os.path.exists(tflite_file)


def test_weights_to_model_file(trained_model, dimension, mdrnn_size):
    weights_file = trained_model["weights_file"]
    print(f"Weights file: {weights_file}")
    tflite_file = tflite_converter.weights_file_to_model_file(weights_file, mdrnn_size, dimension)
    print(f"File returned: {tflite_file}")
    assert os.path.exists(tflite_file)


def test_model_file_to_tflite(trained_model):
    model_filename = trained_model["keras_file"]
    tflite_file = tflite_converter.model_file_to_tflite(model_filename)
    assert os.path.exists(tflite_file)