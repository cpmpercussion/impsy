"""Manages Training Data for the Musical MDN and can generate fake datsets for testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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


class SequenceDataLoader(object):
    """Manages data from a sequence and generates epochs"""

    def __init__(self, num_steps, batch_size, corpus):
        """load corpus and generate examples"""
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.corpus = corpus
        self.examples = self.setup_training_examples()
        print("Done initialising loader.")

    def setup_training_examples(self):
        xs = []
        for i in range(len(self.corpus) - self.num_steps - 1):
            example = self.corpus[i: i + self.num_steps]
            xs.append(example)
        print("Total training examples:", str(len(xs)))
        return xs

    def next_epoch(self):
        """Return an epoch of batches of shuffled examples."""
        np.random.shuffle(self.examples)
        batches = []
        for i in range(len(self.examples) // self.batch_size):
            batch = self.examples[i * self.batch_size: (i + 1) * self.batch_size]
            batches.append(batch)
        return(np.array(batches))
