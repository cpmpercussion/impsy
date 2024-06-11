import click
import time
from .utils import mdrnn_config
import pandas as pd


pd.set_option('display.float_format', lambda x: '%.2f' % x)


@click.command(name="test-mdrnn")
def test_mdrnn():
  """This command simply loads the MDRNN to test that it works and how long it takes."""
  # import tensorflow, do this now to make CLI more responsive.
  print("Importing MDRNN.")
  start_import = time.time()
  import impsy.mdrnn as mdrnn
  import tensorflow.compat.v1 as tf
  print("Done. That took", time.time() - start_import, "seconds.")

  model_config = mdrnn_config("s")

  def build_network(sess, dimension, units, mixes, layers):
    """Build the MDRNN."""
    mdrnn.MODEL_DIR = "./models/"
    tf.keras.backend.set_session(sess)
    with compute_graph.as_default():
        net = mdrnn.PredictiveMusicMDRNN(mode=mdrnn.NET_MODE_RUN,
                                            dimension=dimension,
                                            n_hidden_units=units,
                                            n_mixtures=mixes,
                                            layers=layers)
    return net

  start_build = time.time()
  compute_graph = tf.Graph()
  with compute_graph.as_default():
      sess = tf.Session()
  build_network(sess, 4, model_config["units"], model_config["mixes"], model_config["layers"])
  print("Done. That took", time.time() - start_build, "seconds.")


@click.command(name="test-speed")
def prediction_speed_test():
    """This command runs a speed test experiment with different sized MDRNN models. The output is written to a CSV file."""
    start_import = time.time()
    import impsy.mdrnn as mdrnn
    import tensorflow.compat.v1 as tf
    print("Done. That took", time.time() - start_import, "seconds.")

    def build_network(sess, compute_graph, net_config):
        """Build the MDRNN."""
        mdrnn.MODEL_DIR = "./models/"
        tf.keras.backend.set_session(sess)
        with compute_graph.as_default():
            net = mdrnn.PredictiveMusicMDRNN(mode=mdrnn.NET_MODE_RUN,
                                                dimension=net_config['dimension'],
                                                n_hidden_units=net_config['units'],
                                                n_mixtures=net_config['mixes'],
                                                layers=net_config['layers'])
        return net

    def request_rnn_prediction(input_value, net):
        """ Accesses a single prediction from the RNN. """
        start = time.time()
        output_value = net.generate_touch(input_value)
        time_delta = time.time() - start
        return output_value, time_delta

    def run_test(tests, config):
        times = []
        compute_graph = tf.Graph()
        with compute_graph.as_default():
            sess = tf.Session()
        net = build_network(sess, compute_graph, config)
        for i in range(tests):
            ## Predictions.
            item = mdrnn.random_sample(out_dim=config['dimension'])
            tf.keras.backend.set_session(sess)
            with compute_graph.as_default():
                rnn_output, t = request_rnn_prediction(item, net)
            out_dict = {
                'time': t, 
                'mixes': config['mixes'],
                'layers': config['layers'],
                'units': config['units'],
                'dimension': config['dimension']}
            times.append(out_dict)
        # clean up
        tf.keras.backend.clear_session()
        sess.close()
        return times

    experiments = []
    mdrnn_units = [64, 128, 256, 512]
    dimensions = [2, 3, 4, 5, 6, 7, 8, 9]
    for un in mdrnn_units:
        for dim in dimensions:
            net_config = {"mixes": 5, "layers": 2, "units": un, "dimension": dim}
            times = run_test(100, net_config)
            experiments.extend(times)
    total_experiment = pd.DataFrame.from_records(experiments)
    total_experiment.to_csv("total_exp.csv")
    print(total_experiment.describe())
