#!/usr/local/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import argparse

print("Script to generate a dataset from .log files in the log directory.")

# Input and output to serial are bytes (0-255)
# Output to Pd is a float (0-1)
parser = argparse.ArgumentParser(description='Script to generate a dataset from .log files in the log directory.')
parser.add_argument('-d', '--dimension', type=int, dest='dimension', default=2,
                    help='The dimension of the data to model, must be >= 2.')
parser.add_argument('-s', '--source', dest='sourcedir', default='logs',
                    help='The source directory to obtain .log files')
args = parser.parse_args()


def transform_log_to_sequence_example(logfile, dimension):
    data_names = ['x'+str(i) for i in range(dimension-1)]
    column_names = ['date', 'source'] + data_names
    perf_df = pd.read_csv(logfile,
                          header=None, parse_dates=True,
                          index_col=0, names=column_names)
    #  Filter out RNN lines, just keep 'interface'
    perf_df = perf_df[perf_df.source == 'interface']
    #  Process times.
    perf_df['t'] = perf_df.index
    perf_df.t = perf_df.t.diff()
    perf_df.t = perf_df.t.dt.total_seconds()
    perf_df = perf_df.dropna()
    return np.array(perf_df[['t']+data_names])


# Load up the performances
log_location = "logs/"
log_file_ending = "-" + str(args.dimension) + "d-mdrnn.log"
log_arrays = []

for local_file in os.listdir(log_location):
    if local_file.endswith(log_file_ending):
        print("Processing:", local_file)
        try:
            log = transform_log_to_sequence_example(log_location + local_file,
                                                args.dimension)
            log_arrays.append(log)
        except Exception:
            print("Processing failed for", local_file)        

# Save Performance Data in a compressed numpy file.
dataset_location = 'datasets/'
dataset_filename = 'training-dataset-' + str(args.dimension) + 'd.npz'

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
    raw = l.astype('float32')  # dt, x_1, ... , x_n
    raw_perfs.append(raw)

print("total number of values:", acc)
print("total number of interactions:", interactions)
print("total time represented:", time)
print("total number of perfs in raw array:", len(raw_perfs))
raw_perfs = np.array(raw_perfs)
np.savez_compressed(dataset_location + dataset_filename, perfs=raw_perfs)
print("done saving:", dataset_location + dataset_filename)
