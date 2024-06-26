"""impsy.dataset: functions for generating a dataset from .log files in the log directory."""

import numpy as np
import pandas as pd
import os
import click


def transform_log_to_sequence_example(logfile: str, dimension: int):
    data_names = ["x" + str(i) for i in range(dimension - 1)]
    column_names = ["date", "source"] + data_names
    perf_df = pd.read_csv(
        logfile, header=None, parse_dates=True, index_col=0, names=column_names
    )
    #  Filter out RNN lines, just keep 'interface'
    perf_df = perf_df[perf_df.source == "interface"]
    #  Process times.
    perf_df["t"] = perf_df.index
    perf_df.t = perf_df.t.diff()
    perf_df.t = perf_df.t.dt.total_seconds()
    perf_df = perf_df.dropna()
    return np.array(perf_df[["t"] + data_names])


def generate_dataset(
    dimension: int, source: str = "logs", destination: str = "datasets"
):
    """Generate a dataset from .log files in the log directory."""
    # Load up the performances
    log_location = f"{source}/"
    log_file_ending = "-" + str(dimension) + "d-mdrnn.log"
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
    dataset_location = destination + "/"
    dataset_filename = "training-dataset-" + str(dimension) + "d.npz"

    # Input format is:
    # 0. 1. 2. ... n.
    # dt x1 x2 ... xn

    raw_perfs = []

    acc = 0
    time = 0
    interactions = 0
    for l in log_arrays:
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
    raw_perfs = np.array(raw_perfs)
    np.savez_compressed(dataset_location + dataset_filename, perfs=raw_perfs)
    click.secho("done saving: {dataset_location + dataset_filename}", fg="green")
    return dataset_filename


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
