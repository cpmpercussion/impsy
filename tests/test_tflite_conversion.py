from impsy import tflite_converter
import os

### tflite conversion tests


def test_config_to_tflite(models_location):
    test_config = "configs/default.toml"
    tflite_file = tflite_converter.config_to_tflite(
        test_config, save_path=models_location
    )
    assert os.path.exists(tflite_file)


def test_weights_to_model_file(trained_model, dimension, mdrnn_size):
    weights_file = trained_model["weights_file"]
    print(f"Weights file: {weights_file}")
    tflite_file = tflite_converter.weights_file_to_model_file(
        weights_file, mdrnn_size, dimension
    )
    print(f"File returned: {tflite_file}")
    assert os.path.exists(tflite_file)


def test_model_file_to_tflite(trained_model):
    model_filename = trained_model["keras_file"]
    tflite_file = tflite_converter.model_file_to_tflite(model_filename)
    assert os.path.exists(tflite_file)


def test_model_file_to_tflite_optimised(trained_model, models_location):
    """Test TFLite conversion with optimisation flag enabled."""
    model_filename = trained_model["keras_file"]
    tflite_file = tflite_converter.model_file_to_tflite(
        model_filename, save_path=models_location, optimise=True
    )
    assert os.path.exists(tflite_file)
    # Optimised file should exist and be smaller or equal to non-optimised
    non_opt_file = tflite_converter.model_file_to_tflite(
        model_filename, save_path=models_location
    )
    assert (
        os.path.getsize(tflite_file) <= os.path.getsize(non_opt_file) + 1024
    )  # allow small variance
