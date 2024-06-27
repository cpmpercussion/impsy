from impsy import tflite_converter
import os


def test_config_to_tflite():
    test_config = "configs/default.toml"
    expected_output_path = "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
    tflite_converter.config_to_tflite(test_config)
    assert os.path.exists(expected_output_path)
    os.remove(expected_output_path)


# def test_keras_modfel_to_tflite():
#     test_model = "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.keras"
#     expected_output_path = "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"