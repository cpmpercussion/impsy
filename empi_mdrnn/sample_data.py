"""Manages Training Data for the Musical MDN and can generate fake datsets for testing."""
import numpy as np
import pandas as pd


def generate_data():
    """Generating some Slightly fuzzy sine wave data."""
    NSAMPLE = 50000
    print("Generating", str(NSAMPLE), "toy data samples.")
    t_data = np.float32(np.array(range(NSAMPLE)) / 10.0)
    t_interval = t_data[1] - t_data[0]
    t_r_data = np.random.normal(0, t_interval / 20.0, size=NSAMPLE)
    t_data = t_data + t_r_data
    r_data = np.random.normal(size=NSAMPLE)
    x_data = np.sin(t_data) * 1.0 + (r_data * 0.05)
    df = pd.DataFrame({'t': t_data, 'x': x_data})
    df.t = df.t.diff()
    df.t = df.t.fillna(1e-4)
    print(df.describe())
    return np.array(df)


def random_touch():
    """ Returns a random touch in mixture_mdn format: (dt, x)."""
    return np.array([(0.01 + (np.random.rand() - 0.5) * 0.005), np.random.rand()])


def batch_generator(seq_len, batch_size, dim, corpus):
    """Returns a generator to cut up datasets into batches of features and labels."""
    # Create empty arrays to contain batch of features and labels#
    # # Produce the generator for training
    # generator = batch_generator(SEQ_LEN, BATCH_SIZE, 3, corpus)
    batch_X = np.zeros((batch_size, seq_len, dim))
    batch_y = np.zeros((batch_size, dim))
    while True:
        for i in range(batch_size):
            # choose random example
            l = random.choice(corpus)
            last_index = len(l) - seq_len - 1
            start_index = np.random.randint(0, high=last_index)
            batch_X[i] = l[start_index:start_index+seq_len]
            batch_y[i] = l[start_index+1:start_index+seq_len+1]  # .reshape(1,dim)
        yield batch_X, batch_y


# Functions for slicing up data
def slice_sequence_examples(sequence, num_steps, step_size=1):
    """ Slices a sequence into examples of length num_steps with step size step_size."""
    xs = []
    for i in range((len(sequence) - num_steps) // step_size + 1):
        example = sequence[(i * step_size): (i * step_size) + num_steps]
        xs.append(example)
    return xs


def seq_to_overlapping_format(examples):
    """Takes sequences of seq_len+1 and returns overlapping
    sequences of seq_len."""
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[1:])
    return (xs, ys)


def seq_to_singleton_format(examples):
    """Return the examples in seq to singleton format.
    """
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs, ys)


def generate_synthetic_3D_data():
    """
    Generates some slightly fuzzy sine wave data in through dimensions (plus time).
    """
    NSAMPLE = 50000
    print("Generating", str(NSAMPLE), "toy data samples.")
    t_data = np.float32(np.array(range(NSAMPLE)) / 10.0)
    t_interval = t_data[1] - t_data[0]
    t_r_data = np.random.normal(0, t_interval / 20.0, size=NSAMPLE)
    t_data = t_data + t_r_data
    r_data = np.random.normal(size=NSAMPLE)
    x_data = (np.sin(t_data) + (r_data / 10.0) + 1) / 2.0
    y_data = (np.sin(t_data * 3.0) + (r_data / 10.0) + 1) / 2.0
    df = pd.DataFrame({'a': x_data, 'b': y_data, 't': t_data})
    df.t = df.t.diff()
    df.t = df.t.fillna(1e-4)
    print(df.describe())
    return np.array(df)

