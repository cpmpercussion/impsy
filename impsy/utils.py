import numpy as np
import pandas as pd
import random


# MDRNN config


SIZE_TO_PARAMETERS = {
    "xxs": {
        "units": 16,
        "mixes": 5,
        "layers": 2,
    },
    "xs": {
        "units": 32,
        "mixes": 5,
        "layers": 2,
    },
    "s": {"units": 64, "mixes": 5, "layers": 2},
    "m": {"units": 128, "mixes": 5, "layers": 2},
    "l": {"units": 256, "mixes": 5, "layers": 2},
    "xl": {"units": 512, "mixes": 5, "layers": 3},
    "default": {"units": 128, "mixes": 5, "layers": 2},
}


def mdrnn_config(size: str):
    """Get a config dictionary from a size string as used in the IMPS command line interface."""
    return SIZE_TO_PARAMETERS[size]


# Fake data generator for tests.


def fuzzy_sine_function(t, scale=1.0, fuzz_factor=0.05):
    """A fuzzy sine function with variable fuzz factor"""
    return np.sin(t) * scale + (np.random.normal() * fuzz_factor)


def generate_data(samples: int = 50000, dimension: int = 2):
    """Generating some Slightly fuzzy sine wave data."""
    assert dimension > 1, "dimension must be greater than 1"
    NSAMPLE = samples
    print("Generating", str(NSAMPLE), "toy data samples.")
    t_data = np.float32(np.array(range(NSAMPLE)) / 10.0)
    t_interval = t_data[1] - t_data[0]
    t_r_data = np.random.normal(
        0, t_interval / 20.0, size=NSAMPLE
    )  ## fuzz up the time sampling
    t_data = t_data + t_r_data
    r_data = np.random.normal(size=NSAMPLE)
    # x_data = np.sin(t_data) * 1.0 + (r_data * 0.05)
    df = pd.DataFrame({"t": t_data})
    for i in range(dimension - 1):
        df[f"x{i}"] = df["t"].apply(fuzzy_sine_function, scale=i)
    df.t = df.t.diff()
    df.t = df.t.fillna(1e-4)
    print(df.describe())
    return np.array(df)
