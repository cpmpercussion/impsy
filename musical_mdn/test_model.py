"""Test the Musical MDN Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from . import ed_mixture
import time


# Training Test
# Train on sequences of length 121 with batch size 100.
def test_training():
    x_t_log = generate_data()
    loader = SequenceDataLoader(num_steps=121, batch_size=100, corpus=x_t_log)
    net = TinyJamNet2D(mode=NET_MODE_TRAIN, n_hidden_units=128, n_mixtures=10, batch_size=100, sequence_length=120)
    losses = net.train(loader, 30, saving=True)
    print(losses)
    # Plot the losses.


# Evaluation Test:
# Predict 10000 Datapoints.
def test_evaluation():
    net = TinyJamNet2D(mode=NET_MODE_RUN, n_hidden_units=128, n_mixtures=10, batch_size=1, sequence_length=1)
    first_touch = np.array([0.001, 15.01]).reshape((1, 1, 2))
    with tf.Session() as sess:
        perf = net.generate_performance(first_touch, 10000, sess)
    perf_df = pd.DataFrame({'t': perf.T[0], 'x': perf.T[1]})
    perf_df['time'] = perf_df.t.cumsum()
    plt.show(perf_df.plot('time', 'x', kind='scatter'))
    print(perf_df.describe())
    # Investigate Output
    window = 100
    for n in [1000, 2000, 3000, 4000, 5000, 6000]:
        print("Window:", str(n), 'to', str(n + window))
        plt.plot(perf_df[n:n + window].time, perf_df[n:n + window].x, '.r-')
        plt.show()

if __name__ == "__main__":
    test_training()
