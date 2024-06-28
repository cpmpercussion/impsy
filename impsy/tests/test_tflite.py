from impsy import tflite_converter
import os


def test_config_to_tflite():
    test_config = "configs/default.toml"
    expected_output_path = (
        "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
    )
    tflite_converter.config_to_tflite(test_config)
    assert os.path.exists(expected_output_path)
    os.remove(expected_output_path)


def test_weights_to_model_file():
    location = "tests/models"
    os.makedirs(location, exist_ok=True)
    dimension = 9
    weights_file = "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.h5"
    size = "s"
    tflite_converter.weights_file_to_model_file(weights_file, size, dimension, location)


def test_model_file_to_tflite():
    model_filename = "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.keras"
    expected_output_path = (
        "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
    )
    tflite_converter.model_file_to_tflite(model_filename)
    assert os.path.exists(expected_output_path)
    os.remove(expected_output_path)
