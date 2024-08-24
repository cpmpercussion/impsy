"""
Experiment to compare inference speeds between keras and tflite models.
24 Aug 2024.
"""

import numpy as np
from impsy import mdrnn
from impsy import tflite_converter
import time
from pathlib import Path
import pandas as pd
import click
import os


np.set_printoptions(precision=2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)


location = Path("experiment_data")
os.makedirs(location, exist_ok=True)


def create_models(dimension=4, units=64, mixes=5, layers=2):
    """Builds a network and saves weights, model file, and tflite file."""
    model = mdrnn.build_mdrnn_model(dimension, units, mixes, layers, inference=True, seq_length=1)
    model_name = mdrnn.mdrnn_model_name(dimension, layers, units, mixes)
    # weights_file = location / f"{model_name}.h5"
    model_file = location / f"{model_name}.keras"
    # model.save_weights(weights_file)
    model.save(model_file)
    tflite_file = tflite_converter.model_to_tflite(model, model_file)

    output = {
        "dimension": dimension,
        "units": units,
        "mixes": mixes,
        "layers": layers,
        "model": model,
        # "weights": weights_file,
        "keras": model_file,
        "tflite": tflite_file
    }

    return output


def experiment(inference_model, model_type, config, num_tests):
    """run the experiment with a specific model."""
    times = []
    for i in range(num_tests):
        input_value = mdrnn.random_sample(out_dim=config["dimension"])
        start = time.time()
        inference_model.generate(input_value)
        time_delta = time.time() - start
        if i > 0:
          # omit _first_ prediction which is a bit bigger due to setup.
          out_dict = {
              "time": time_delta * 1000, # use ms
              "mixes": config["mixes"],
              "layers": config["layers"],
              "units": config["units"],
              "dimension": config["dimension"],
              "model_type": model_type,
          }
          times.append(out_dict)
    return times


def run_test(num_tests, config):
    """runs the experiment for each model under test"""
    times = []
    dim = config["dimension"]
    units = config["units"]
    mixes = config["mixes"]
    layers = config["layers"]

    models = create_models(dim, units, mixes, layers)
    keras_model = mdrnn.KerasMDRNN(models["keras"],dim, units, mixes, layers)
    times +=experiment(keras_model, "keras", config, num_tests)
    tflite_model = mdrnn.TfliteMDRNN(models["tflite"],dim, units, mixes, layers)
    times += experiment(tflite_model, "tflite", config, num_tests)

    # times += experiment(h5_model, "h5", config, num_tests)
    # h5_model = mdrnn.KerasMDRNN(models["weights"],dim, units, mixes, layers)
    # times += experiment(dummy_model, "dummy", config, num_tests)
    # dummy_model = mdrnn.DummyMDRNN(location, dim, units, mixes, layers)

    model_files = [models["keras"], models["tflite"]]
    return times, model_files

### Setup and start the experiment.
# parameter combinations
mdrnn_units = [64, 128, 256, 512]
dimensions = [2, 4, 6, 8, 10]
number_of_tests = 100

model_files = []
experiments = []

for un in mdrnn_units:
    for dim in dimensions:
        net_config = {"mixes": 5, "layers": 2, "units": un, "dimension": dim}
        times, files = run_test(number_of_tests + 1, net_config)
        experiments.extend(times)
        model_files.extend(files)
total_experiment = pd.DataFrame.from_records(experiments)
total_experiment.to_csv(location / "total_exp.csv")

# click.secho("All experiment data:", fg="green")
# click.secho(total_experiment.describe())

# click.secho("h5 experiment data:", fg="green")
# click.secho(total_experiment[total_experiment['model_type'] == 'h5'].describe())

click.secho("keras experiment data:", fg="green")
click.secho(total_experiment[total_experiment['model_type'] == 'keras'].describe())

click.secho("tflite experiment data:", fg="green")
click.secho(total_experiment[total_experiment['model_type'] == 'tflite'].describe())

## delete all the model files created.
for model_file in model_files:
    os.remove(model_file)
