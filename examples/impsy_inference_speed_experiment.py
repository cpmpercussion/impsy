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
pd.set_option("display.float_format", lambda x: "%.3f" % x)


location = Path("experiment_data")
os.makedirs(location, exist_ok=True)


def create_models(dimension=4, units=64, mixes=5, layers=2):
    """Builds a network and saves weights, model file, and tflite file."""
    start_model_build = time.time()
    model = mdrnn.build_mdrnn_model(dimension, units, mixes, layers, inference=True, seq_length=1)
    model_build_time = time.time() - start_model_build
    model_name = mdrnn.mdrnn_model_name(dimension, layers, units, mixes)

    click.secho(f"Built {model_name} in {model_build_time:.3f}s")

    model_path = location / f"{model_name}.keras"
    tflite_path = location / f"{model_name}.tflite"

    ## Only bother creating if the files don't exist.
    if not model_path.exists():
        model.save(model_path)
    
    if not tflite_path.exists():
        tflite_path = tflite_converter.model_to_tflite(model, model_path)

    output = {
        "dimension": dimension,
        "units": units,
        "mixes": mixes,
        "layers": layers,
        "name": model_name,
        "model": model,
        "keras": model_path,
        "tflite": tflite_path,
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
    inference_times = []
    load_times = []
    dim = config["dimension"]
    units = config["units"]
    mixes = config["mixes"]
    layers = config["layers"]

    models = create_models(dim, units, mixes, layers)
    start_model_load = time.time()
    keras_model = mdrnn.KerasMDRNN(models["keras"],dim, units, mixes, layers)
    keras_load_time = time.time() - start_model_load

    inference_times += experiment(keras_model, "keras", config, num_tests)

    start_model_load = time.time()
    tflite_model = mdrnn.TfliteMDRNN(models["tflite"],dim, units, mixes, layers)
    tflite_load_time = time.time() - start_model_load

    inference_times += experiment(tflite_model, "tflite", config, num_tests)

    load_times.append({
        "keras_load": keras_load_time, # use s
        "tflite_load": tflite_load_time, # use s
        "mixes": mixes,
        "layers": layers,
        "units": units,
        "dimension": dim,
    })

    model_files = [models["keras"], models["tflite"]]
    return inference_times, model_files, load_times

### Setup and start the experiment.
# parameter combinations
mdrnn_units = [64, 128, 256, 512]
dimensions = [2, 4, 6, 8, 10]
number_of_tests = 100

model_files = []
exp_inference_times = []
exp_load_times = []

for un in mdrnn_units:
    for dim in dimensions:
        net_config = {"mixes": 5, "layers": 2, "units": un, "dimension": dim}
        inference_times, files, load_times = run_test(number_of_tests + 1, net_config)
        exp_inference_times.extend(inference_times)
        exp_load_times.extend(load_times)
        model_files.extend(files)
inference_experiment = pd.DataFrame.from_records(exp_inference_times)
inference_experiment.to_csv(location / "impsy_experiment_inference.csv")
load_experiment = pd.DataFrame.from_records(exp_load_times)
load_experiment.to_csv(location / "impsy_experiment_loads.csv")

click.secho("keras experiment data:", fg="green")
click.secho(inference_experiment[inference_experiment['model_type'] == 'keras'].describe())

click.secho("tflite experiment data:", fg="green")
click.secho(inference_experiment[inference_experiment['model_type'] == 'tflite'].describe())

click.secho("model load data:", fg="green")
click.secho(load_experiment.describe())

## delete all the model files created.
# for model_file in model_files:
#     os.remove(model_file)
