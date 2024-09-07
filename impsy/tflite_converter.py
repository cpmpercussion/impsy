"""impsy.tflite_converter: Functions for converting a model to tflite format."""

import click
from .utils import mdrnn_config, get_config_data
from pathlib import Path


def model_to_tflite(model, model_path: Path, save_path: Path = None, optimise=False):
    """This actually converts a loaded Keras model to tflite format."""
    import tensorflow as tf

    # Setup output path and name.
    output_file = model_path.with_suffix(".tflite")
    if save_path is not None:
        output_file = save_path / output_file.name

    click.secho("Setup converter.", fg="blue")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    if optimise:
        click.secho("Using default optimisations: this will reduce model size but may degrade performance.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    click.secho("Do the conversion.", fg="blue")
    tflite_model = converter.convert()
    
    click.secho("Print Analysis...", fg="blue")
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

    click.secho("Saving..", fg="blue")
    click.secho(f"Saving tflite model to: {output_file}", fg="blue")
    with open(output_file, "wb") as f:
        f.write(tflite_model)
    return output_file


def model_file_to_tflite(filename, save_path = None, optimise=False):
    """Converts a given model"""
    import tensorflow as tf
    import keras_mdn_layer as mdn_layer

    model_file = Path(filename)
    assert model_file.suffix == ".keras", "This function only works on .keras files."
    loaded_model = tf.keras.saving.load_model(
        filename, custom_objects={"MDN": mdn_layer.MDN}
    )
    tflite_file = model_to_tflite(loaded_model, model_file, save_path=save_path, optimise=optimise)
    return tflite_file



def config_to_tflite(config_path, save_path = None, optimise=False):
    """Converts the model specified in a config dictionary to tflite format."""
    import tensorflow as tf
    import impsy.mdrnn as mdrnn

    click.secho("IMPSY: Converting model to tflite.", fg="blue")
    config = get_config_data(config_path)

    click.secho(f"MDRNN: Using {config['model']['size']} model.", fg="green")

    model_config = mdrnn_config(config["model"]["size"])
    net = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_RUN,
        dimension=config["model"]["dimension"],
        n_hidden_units=model_config["units"],
        n_mixtures=model_config["mixes"],
        layers=model_config["layers"],
    )
    click.secho(f"MDRNN Loaded: {net.model_name}", fg="green")
    model_path = Path(config["model"]["file"])
    net.load_model(model_file=model_path)
    tflite_file = model_to_tflite(net.model, model_path, save_path, optimise=optimise)
    return tflite_file


def weights_file_to_model_file(weights_file, model_size, dimension, save_path = None):
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
    keras_file_path = Path(weights_file).with_suffix(".keras")
    if save_path is not None:
        keras_file_path = Path(save_path) / keras_file_path.name
    inference_model.model.save(keras_file_path)
    return keras_file_path


@click.command(name="convert-tflite")
@click.option('--model', '-m', help='Path to a .keras model or .h5 weights')
@click.option('--dimension', '-d', type=int, help='Dimension (only needed for h5 files)')
@click.option('--size', '-s', help="Size, one of xs, s, m, l, (only needed for h5 files)")
@click.option('--out_dir', '-o', help="Output location for tflite file.")
@click.option(
    "--optimise/--no-optimise", default=False, help="Use default optimisations in TFLite conversion (may degrade model performance, but reduce model size)."
)
def convert_tflite(model, dimension, size, out_dir, optimise):
    """Convert existing IMPSY model to tflite format."""
    if model is None:
        config_to_tflite("config.toml", save_path=out_dir, optimise=optimise)
    elif Path(model).suffix == ".keras":
        # it's a keras file
        model_file_to_tflite(model, save_path=out_dir, optimise=optimise)
    elif Path(model).suffix == ".h5":
        # it's an h5 file
        if dimension is not None and size is not None:
            model_file = weights_file_to_model_file(model, size, dimension, save_path=out_dir)
            model_file_to_tflite(model_file, save_path=out_dir, optimise=optimise)
        else:
            click.secho("You need to specify a dimension and size to convert an h5 file.")
