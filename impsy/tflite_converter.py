"""impsy.tflite_converter: Functions for converting a model to tflite format."""

import click
from .utils import mdrnn_config
import tomllib


def build_network(config):
    """Build the MDRNN, uses a high-level size parameter and dimension."""
    return net

def model_to_tflite(model, output_name):
    import tensorflow as tf

    click.secho("Setup converter.")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    click.secho("Do the conversion.")
    tflite_model = converter.convert()

    click.secho("Saving..")
    tflite_model_name = f'{output_name}.tflite'
    with open(tflite_model_name, "wb") as f:
        f.write(tflite_model)


def model_file_to_tflite(filename):
    """Converts a given model """
    import tensorflow as tf

    assert filename[-6:] == ".keras", "This function only works on .keras files."
    loaded_model = tf.keras.saving.load_model(filename)
    name = f[:6]
    model_to_tflite(loaded_model, name)


def config_to_tflite(config_path):
    """Converts the model specified in a config dictionary to tflite format."""
    import tensorflow as tf
    import impsy.mdrnn as mdrnn

    click.secho("IMPSY: Converting model to tflite.", fg="blue")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    click.secho(f"MDRNN: Using {config['model']['size']} model.", fg="green")
    model_config = mdrnn_config(config["model"]["size"])
    net = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_RUN,
        dimension=config["model"]["dimension"],
        n_hidden_units=model_config["units"],
        n_mixtures=model_config["mixes"],
        layers=model_config["layers"],
    )
    click.secho(f"MDRNN Loaded: {net.model_name()}", fg="green")
    net.load_model(model_file=config["model"]["file"])
    output_name = config["model"]["file"].removesuffix(".h5")
    output_name = output_name.removesuffix(".keras")
    model_to_tflite(net.model, output_name)


@click.command(name="convert-tflite")
def convert_tflite():
    """Convert existing IMPSY model to tflite format."""
    config_to_tflite("config.toml")
