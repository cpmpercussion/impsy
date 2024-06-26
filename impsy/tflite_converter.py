"""impsy.tflite_converter: Functions for converting a model to tflite format."""

import click
from .utils import mdrnn_config
import tomllib


def build_network(config):
    """Build the MDRNN, uses a high-level size parameter and dimension."""
    import impsy.mdrnn as mdrnn

    click.secho(f"MDRNN: Using {config['model']['size']} model.", fg="green")
    model_config = mdrnn_config(config["model"]["size"])
    mdrnn.MODEL_DIR = "./models/"
    net = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_RUN,
        dimension=config["model"]["dimension"],
        n_hidden_units=model_config["units"],
        n_mixtures=model_config["mixes"],
        layers=model_config["layers"],
    )
    net.pi_temp = config["model"]["pitemp"]
    net.sigma_temp = config["model"]["sigmatemp"]
    click.secho(f"MDRNN Loaded: {net.model_name()}", fg="green")
    return net


@click.command(name="convert-tflite")
def convert_tflite():
    """Convert existing IMPSY model to tflite format."""
    import tensorflow as tf

    click.secho("IMPSY: Converting model to tflite.", fg="blue")
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    net = build_network(config)
    net.load_model(model_file=config["model"]["file"])
    # setup converter
    click.secho("Setup converter.")
    converter = tf.lite.TFLiteConverter.from_keras_model(net.model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    click.secho("Do the conversion.")
    tflite_model = converter.convert()

    click.secho("Saving..")
    tflite_model_name = f'{config["model"]["file"]}-lite.tflite'
    with open(tflite_model_name, "wb") as f:
        f.write(tflite_model)
