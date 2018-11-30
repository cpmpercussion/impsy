from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
from . import mixture_2d_normals


tf.logging.set_verbosity(tf.logging.INFO)  # set logging.


NET_MODE_TRAIN = 'train'
NET_MODE_RUN = 'run'
MDN_MODEL_TENSORFLOW = 'tf'
MDN_MODEL_SKETCH = 'sketch'
MODEL_DIR = "./"
LOG_PATH = "./output-logs/"

def proc_generated_touch(touch):
    """ Processes a generated touch to have dt > 0, and 0 <= x <= 1 """
    dt = max(touch[0], 0.000454)
    x_loc = min(max(touch[1], 0), 1)
    return np.array([dt, x_loc])


class MixtureRNN(object):
    """Mixture Density Network RNN using the SketchRNN's hand-coded loss function for the mixture of 2D Normals."""

    def __init__(self, mode=NET_MODE_TRAIN, n_hidden_units=128, n_mixtures=24, batch_size=100, sequence_length=120, layers=1):
        """Initialise the TinyJamNet model. Use mode='run' for evaluation graph and mode='train' for training graph."""
        # hyperparameters
        self.mode = mode
        self.n_hidden_units = n_hidden_units
        self.n_rnn_layers = layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.st_dev = 0.5
        self.n_mixtures = n_mixtures  # number of mixtures
        self.n_input_units = 2  # Number of dimensions of the input (and sampled output) data
        self.mdn_splits = 6  # (pi, sigma_1, sigma_2, mu_1, mu_2, rho)
        self.n_output_units = n_mixtures * self.mdn_splits  # KMIX * self.mdn_splits
        self.lr = 1e-4  # could be 1e-3
        self.lr_decay_rate = 0.9999,  # Learning rate decay per minibatch.
        self.lr_minimum = 0.00001,  # Minimum learning rate.
        self.grad_clip = 1.0
        self.state = None
        self.use_input_dropout = False
        if self.mode is NET_MODE_TRAIN:
            self.use_input_dropout = True
        self.dropout_prob = 0.90
        self.run_name = self.get_run_name()

        tf.reset_default_graph()
        self.graph = tf.get_default_graph()

        # Construct Operation Graph
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.sequence_length, self.n_input_units], name="x")  # input
                self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.sequence_length, self.n_input_units], name="y")  # target
                self.rnn_outputs, self.init_state, self.final_state = self.recurrent_network(self.x)

            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.n_hidden_units], name="reshape_rnn_outputs")
            output_params = self.fully_connected_layer(self.rnn_outputs, self.n_hidden_units, self.n_output_units)
            self.pis, self.scales_1, self.scales_2, self.locs_1, self.locs_2, self.corr = mixture_2d_normals.split_tensor_to_mixture_parameters(output_params)
            self.saver = tf.train.Saver(name="saver")  # Saver

            if self.mode is NET_MODE_TRAIN:  # Loading training operations.
                tf.logging.info("Loading Training Operations")
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                with tf.name_scope('labels'):
                    y_reshaped = tf.reshape(self.y, [-1, self.n_input_units], name="reshape_labels")
                    [y1_data, y2_data] = tf.split(y_reshaped, 2, 1)
                loss_func = mixture_2d_normals.get_lossfunc(self.pis, self.locs_1, self.locs_2, self.scales_1, self.scales_2, self.corr, y1_data, y2_data)
                self.cost = tf.reduce_mean(loss_func)
                optimizer = tf.train.AdamOptimizer(self.lr)
                gvs = optimizer.compute_gradients(self.cost)
                capped_gvs = [(tf.clip_by_value(grad, -1 * self.grad_clip, self.grad_clip), var) for grad, var in gvs]
                self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')
                self.training_state = None
                tf.summary.scalar("cost_summary", self.cost)
                # Summaries
                self.summaries = tf.summary.merge_all()

            if self.mode is NET_MODE_RUN:  # Loading running operations? None needed probably.
                tf.logging.info("Loading Running Operations")
                # TODO: write a sketch-RNN version of the sampling function?

        if self.mode is NET_MODE_TRAIN:
            self.writer = tf.summary.FileWriter(LOG_PATH + self.run_name + '/', graph=self.graph)
        train_vars_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        tf.logging.info("done initialising: %s vars: %d", self.model_name(), train_vars_count)

    def fully_connected_layer(self, X, in_dim, out_dim):
        with tf.name_scope('rnn_to_mdn'):
            W = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=self.st_dev, dtype=tf.float32))
            b = tf.Variable(tf.random_normal([1, out_dim], stddev=self.st_dev, dtype=tf.float32))
            output = tf.matmul(X, W) + b
        tf.summary.histogram("out_weights", W)
        tf.summary.histogram("out_biases", b)
        tf.summary.histogram("out_logits", output)
        return output

    def recurrent_network(self, X):
        """ Create the RNN part of the network. """
        with tf.name_scope('recurrent_network'):
            cells_list = [tf.contrib.rnn.LSTMCell(self.n_hidden_units, state_is_tuple=True) for _ in range(self.n_rnn_layers)]
            if self.use_input_dropout:
                cells_list = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_prob) for cell in cells_list]
            cell = tf.contrib.rnn.MultiRNNCell(cells_list, state_is_tuple=True)
            init_state = cell.zero_state(self.batch_size, tf.float32)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(
                cell,
                X,
                initial_state=init_state,
                time_major=False,
                dtype=tf.float32,
                scope='RNN'
            )
            #final_state = tf.identity(final_state, name="state") # recurrent_network/state
        return rnn_outputs, init_state, final_state

    def model_name(self):
        """Returns the name of the present model for saving to disk"""
        return "mixture-rnn-" + str(self.n_rnn_layers) + "layers-" + str(self.n_hidden_units) + "units"

    def freeze_model(self, name=""):
        input_graph_def = self.graph.as_graph_def()
        sess = tf.Session()
        self.prepare_model_for_running(sess)
        output_node_names = "2d_mixture_split/logits_2d,2d_mixture_split/scales_2d_1,2d_mixture_split/scales_2d_2,2d_mixture_split/corr_2d,2d_mixture_split/locs_2d_1,2d_mixture_split/locs_2d_2,input/recurrent_network/state"#/input/(input)"
        output_graph = "frozen_model.pb"
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


    def get_run_name(self):
        out = self.model_name() + "-"
        out += time.strftime("%Y%m%d-%H%M%S")
        return out

    def train_batch(self, batch, sess):
        """Train the network on one batch"""
        # batch is an array of shape (batch_size, sequence_length + 1, n_input_units)
        batch_x = batch[:, :self.sequence_length, :]
        batch_y = batch[:, 1:, :]
        feed = {self.x: batch_x, self.y: batch_y}
        if self.training_state is not None:
            feed[self.init_state] = self.training_state
        training_loss_current, self.training_state, _, summary, step = sess.run([self.cost, self.final_state, self.train_op, self.summaries, self.global_step], feed_dict=feed)
        self.writer.add_summary(summary, step)
        return training_loss_current, step

    def train_epoch(self, batches, sess):
        """Train the network on one epoch of training data."""
        total_training_loss = 0
        epoch_steps = 0
        total_steps = len(batches)
        step = 0
        for b in batches:
            training_loss, step = self.train_batch(b, sess)
            epoch_steps += 1
            total_training_loss += training_loss
            if (epoch_steps % 200 == 0):
                tf.logging.info("trained batch: %d of %d; loss was %f", epoch_steps, total_steps, training_loss)
        return (total_training_loss / epoch_steps), step

    def train(self, data_manager, num_epochs, saving=True):
        """Train the network for the a number of epochs."""
        self.num_epochs = num_epochs
        tf.logging.info("going to train: %s", self.model_name())
        start_time = time.time()
        training_losses = []
        step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_epochs):
                batches = data_manager.next_epoch()
                epoch_average_loss, step = self.train_epoch(batches, sess)
                training_losses.append(epoch_average_loss)
                tf.logging.info("trained epoch %d of %d", i, self.num_epochs)
                if saving:
                    checkpoint_path = LOG_PATH + self.run_name + '/' + self.model_name() + ".ckpt"
                    tf.logging.info('saving model %s, global_step %d.', checkpoint_path, step)
                    self.saver.save(sess, checkpoint_path, global_step=step)
            if saving:
                tf.logging.info('saving model %s.', './' + self.model_name())
                self.saver.save(sess, self.model_name())
        tf.logging.info("took %d seconds to train.", (time.time() - start_time))
        return training_losses

    def prepare_model_for_running(self, sess):
        """Load trained model and reset RNN state."""
        sess.run(tf.global_variables_initializer())
        self.saver.restore(sess, MODEL_DIR + self.model_name())
        self.state = None

    def generate_touch(self, prev_touch, sess):
        """Generate prediction for a single touch."""
        input_touch = prev_touch.reshape([1, 1, self.n_input_units])  # Give input correct shape for one-at-a-time evaluation.
        if self.state is not None:
            feed = {self.x: input_touch, self.init_state: self.state}
        else:
            feed = {self.x: input_touch}
        pis, locs_1, locs_2, scales_1, scales_2, corr, self.state = sess.run([self.pis, self.locs_1, self.locs_2, self.scales_1, self.scales_2, self.corr, self.final_state], feed_dict=feed)
        x_1, x_2 = mixture_2d_normals.sample_mixture_model(pis[0], locs_1[0], locs_2[0], scales_1[0], scales_2[0], corr[0], temp=1.0, greedy=False)
        return np.array([x_1, x_2])

    def generate_performance(self, first_touch, number, sess):
        self.prepare_model_for_running(sess)
        previous_touch = first_touch
        performance = [previous_touch.reshape((self.n_input_units,))]
        for i in range(number):
            previous_touch = self.generate_touch(previous_touch, sess)
            previous_touch = proc_generated_touch(previous_touch)
            performance.append(previous_touch.reshape((self.n_input_units,)))
        return np.array(performance)
