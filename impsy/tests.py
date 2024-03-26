import click
import time
from .utils import mdrnn_config


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
    print("MDRNN Loaded.")
    return net

  start_build = time.time()
  compute_graph = tf.Graph()
  with compute_graph.as_default():
      sess = tf.Session()
  build_network(sess, 4, model_config["units"], model_config["mixes"], model_config["layers"])
  print("Done. That took", time.time() - start_build, "seconds.")
