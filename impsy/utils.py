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


# Manages Training Data for the Musical MDN and can generate fake datsets for testing.


def batch_generator(seq_len, batch_size, dim, corpus):
    """Returns a generator to cut up datasets into
    batches of features and labels."""
    # generator = batch_generator(SEQ_LEN, BATCH_SIZE, 3, corpus)
    batch_X = np.zeros((batch_size, seq_len, dim))
    batch_y = np.zeros((batch_size, dim))
    while True:
        for i in range(batch_size):
            # choose random example
            l = random.choice(corpus)
            last_index = len(l) - seq_len - 1
            start_index = np.random.randint(0, high=last_index)
            batch_X[i] = l[start_index : start_index + seq_len]
            batch_y[i] = l[
                start_index + 1 : start_index + seq_len + 1
            ]  # .reshape(1,dim)
        yield batch_X, batch_y


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
    t_r_data = np.random.normal(0, t_interval / 20.0, size=NSAMPLE) ## fuzz up the time sampling
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
