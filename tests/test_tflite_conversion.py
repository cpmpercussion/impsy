from impsy import tflite_converter
import os
import numpy as np

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


# These write to fresh directories rather than next to the session model:
# earlier tests may still have that .tflite loaded, and Windows can't
# overwrite a memory-mapped file.


def test_model_file_to_tflite(trained_model, tmp_path):
    model_filename = trained_model["keras_file"]
    tflite_file = tflite_converter.model_file_to_tflite(
        model_filename, save_path=tmp_path
    )
    assert os.path.exists(tflite_file)


def test_model_file_to_tflite_optimised(trained_model, tmp_path):
    """Test TFLite conversion with optimisation flag enabled."""
    model_filename = trained_model["keras_file"]
    (tmp_path / "optimised").mkdir()
    (tmp_path / "plain").mkdir()
    tflite_file = tflite_converter.model_file_to_tflite(
        model_filename, save_path=tmp_path / "optimised", optimise=True
    )
    assert os.path.exists(tflite_file)
    # Optimised file should exist and be smaller or equal to non-optimised
    non_opt_file = tflite_converter.model_file_to_tflite(
        model_filename, save_path=tmp_path / "plain"
    )
    assert (
        os.path.getsize(tflite_file) <= os.path.getsize(non_opt_file) + 1024
    )  # allow small variance


def test_checkpoint_file_to_tflite(dimension, units, mixtures, layers, tmp_path):
    """Training-model checkpoints (as saved by ModelCheckpoint) should convert too."""
    from impsy import mdrnn

    training_model = mdrnn.build_mdrnn_model(
        dimension, units, mixtures, layers, inference=False
    )
    checkpoint_file = tmp_path / "test-ckpt.keras"
    training_model.save(checkpoint_file)
    tflite_file = tflite_converter.model_file_to_tflite(checkpoint_file)
    assert os.path.exists(tflite_file)
    runner = mdrnn.TfliteMDRNN(tflite_file, dimension, units, mixtures, layers)
    assert len(runner.generate(np.zeros(dimension, dtype=np.float32))) == dimension
