"""impsy.tflite_converter: Functions for converting a model to tflite format."""

import click
from .utils import mdrnn_config
import tomllib


def model_to_tflite(model, output_name: str):
    import tensorflow as tf

    click.secho("Setup converter.", fg="blue")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    click.secho("Do the conversion.", fg="blue")
    tflite_model = converter.convert()

    click.secho("Saving..", fg="blue")
    tflite_model_name = f"{output_name}.tflite"
    click.secho(f"Saving tflite model to: {tflite_model_name}", fg="blue")
    with open(tflite_model_name, "wb") as f:
        f.write(tflite_model)


def model_file_to_tflite(filename):
    """Converts a given model"""
    import tensorflow as tf
    import keras_mdn_layer as mdn_layer

    assert filename[-6:] == ".keras", "This function only works on .keras files."
    loaded_model = tf.keras.saving.load_model(
        filename, custom_objects={"MDN": mdn_layer.MDN}
    )
    file_stem = filename.removesuffix(".keras")
    model_to_tflite(loaded_model, file_stem)


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


def weights_file_to_model_file(weights_file, model_size, dimension, location):
    """Constructs a model from a given weights file and saves as a .keras inference model."""
    import impsy.mdrnn as mdrnn

    model_config = mdrnn_config(model_size)
    inference_model = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_RUN,
        dimension=dimension,
        n_hidden_units=model_config["units"],
        n_mixtures=model_config["mixes"],
        layers=model_config["layers"],
    )
    inference_model.load_model(model_file=weights_file)
    model_name = inference_model.model_name()
    inference_model.model.save(f"{location}/{model_name}.keras")


@click.command(name="convert-tflite")
def convert_tflite():
    """Convert existing IMPSY model to tflite format."""
    config_to_tflite("config.toml")
