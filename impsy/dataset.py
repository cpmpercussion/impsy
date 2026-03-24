"""impsy.dataset: functions for generating a dataset from .log files in the log directory."""

import numpy as np
import csv
import os
import click
from pathlib import Path
from datetime import datetime


def transform_log_to_sequence_example(logfile: str, dimension: int):
    """Transform a log file into a numpy array of (dt, x_1, ..., x_n) sequences."""
    timestamps = []
    values = []
    n_data_cols = dimension - 1
    # remember that logs are in format:
    # timestamp, source, x_1, x_2, ..., x_n

    # this used to be simple pandas, but to eliminate that dependency we do it manually.
    with open(logfile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2 + n_data_cols:
                continue
            source = row[1].strip()
            if source != "interface":
                continue # only take interface interactions
            try:
                ts = datetime.fromisoformat(row[0].strip())
                data = [float(row[i]) for i in range(2, 2 + n_data_cols)]
                timestamps.append(ts)
                values.append(data)
            except (ValueError, IndexError):
                continue

    if len(timestamps) < 2:
        return np.array([]).reshape(0, dimension)

    # Compute time deltas
    result = []
    for i in range(1, len(timestamps)):
        dt = (timestamps[i] - timestamps[i - 1]).total_seconds()
        result.append([dt] + values[i])

    return np.array(result, dtype=np.float64)


def generate_dataset(
    dimension: int, source: str = "logs", destination: str = "datasets"
):
    """Generate a dataset from .log files in the log directory."""
    # Load up the performances
    log_location = f"{source}/"
    log_file_ending = f"-{dimension}d-mdrnn.log"
    log_arrays = []

    for local_file in os.listdir(log_location):
        if local_file.endswith(log_file_ending):
            print("Processing:", local_file)
            try:
                log = transform_log_to_sequence_example(
                    log_location + local_file, dimension
                )
                log_arrays.append(log)
            except Exception:
                print("Processing failed for", local_file)

    # Save Performance Data in a compressed numpy file.
    dataset_name = f"training-dataset-{dimension}d.npz"
    dataset_file = Path(destination) / dataset_name

    # Input format is:
    # 0. 1. 2. ... n.
    # dt x1 x2 ... xn

    raw_perfs = []

    acc = 0
    time = 0
    interactions = 0
    for l in log_arrays:
        if l.shape[0] == 0:
            continue # ignore logs with zero values.
        acc += l.shape[0] * l.shape[1]
        interactions += l.shape[0]
        time += l.T[0].sum()
        raw = l.astype("float32")  # dt, x_1, ... , x_n
        raw_perfs.append(raw)

    if acc == 0:
        click.secho("Zero values to add to dataset! aborting.", fg="red")
        return

    click.secho(f"total number of values: {acc}", fg="blue")
    click.secho(f"total number of interactions: {interactions}", fg="blue")
    click.secho(f"total time represented: {time}", fg="blue")
    click.secho(f"total number of perfs in raw array: {len(raw_perfs)}", fg="blue")
    raw_perfs = np.array(raw_perfs, dtype=object) # use object encoding to allow inhomogeneous arrays.
    np.savez_compressed(dataset_file, perfs=raw_perfs)
    click.secho(f"done saving: {dataset_name}", fg="green")
    return dataset_file


@click.command(name="dataset")
@click.option(
    "-D",
    "--dimension",
    type=int,
    default=2,
    help="The dimension of the data to model, must be >= 2.",
)
@click.option(
    "-S",
    "--source",
    type=str,
    default="logs",
    help="The source directory to obtain .log files.",
)
def dataset(dimension: int, source: str):
    """Generate a dataset from .log files in the log directory."""
    generate_dataset(dimension, source)
