import logging
import time
import datetime
import numpy as np
import pandas as pd

# Hack to get openMP working annoyingly.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print("Importing Keras and MDRNN.")
start_import = time.time()
import empi_mdrnn
import tensorflow as tf
from keras import backend as K
from tensorflow.contrib.training.python.training.hparam import HParams
print("Done. That took", time.time() - start_import, "seconds.")

def build_network(sess, compute_graph, net_config):
    """Build the MDRNN."""
    empi_mdrnn.MODEL_DIR = "./models/"
    K.set_session(sess)
    with compute_graph.as_default():
        net = empi_mdrnn.PredictiveMusicMDRNN(mode=empi_mdrnn.NET_MODE_RUN,
                                              dimension=net_config.dimension,
                                              n_hidden_units=net_config.units,
                                              n_mixtures=net_config.mixes,
                                              layers=net_config.layers)
        #net.pi_temp = net_config.pi_temp
        #net.sigma_temp = net_config.sigmatemp
    print("MDRNN Loaded.")
    return net


def request_rnn_prediction(input_value, net):
    """ Accesses a single prediction from the RNN. """
    start = time.time()
    output_value = net.generate_touch(input_value)
    time_delta = time.time() - start
    #print("Prediction took:", time_delta)
    return output_value, time_delta


def run_test(tests, net_config):
    times = pd.DataFrame()
    compute_graph = tf.Graph()
    with compute_graph.as_default():
        sess = tf.Session()
    net = build_network(sess, compute_graph, net_config)
    for i in range(tests):
        ## Predictions.
        item = empi_mdrnn.random_sample(out_dim=net_config.dimension)
        K.set_session(sess)
        with compute_graph.as_default():
            rnn_output, t = request_rnn_prediction(item, net)
        out_dict = {
            'time': t, 
            'mixes': net_config.mixes,
            'layers': net_config.layers,
            'units': net_config.units,
            'dimension': net_config.dimension}
        times = times.append(out_dict, ignore_index=True)
    # clean up
    K.clear_session()
    sess.close()
    return times


if __name__ == "__main__":
    experiment_frames = []
    # hparams = HParams(mixes=5, layers=2, units=64, dimension=2)
    mdrnn_units = [64, 128, 256, 512]
    dimensions = [2, 3, 4, 5, 6, 7, 8, 9]
    for un in mdrnn_units:
        for dim in dimensions:
            hparams = HParams(mixes=5, layers=2, units=un, dimension=dim)
            times = run_test(100, hparams)
            experiment_frames.append(times)
    total_experiment = pd.concat(experiment_frames, ignore_index=True)
    total_experiment.to_csv("total_exp.csv")
    print(total_experiment.describe())


# sysctl -n machdep.cpu.brand_string
