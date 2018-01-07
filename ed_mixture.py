""" Mixture Model Operations used in Edward Mixture Model included in Tensorflow """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tf.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag


N_MIXTURES = 24
MDN_SPLITS = 5  # (pi, sigma_1, sigma_2, mu_1, mu_2)
N_OUTPUT_UNITS = N_MIXTURES * MDN_SPLITS


def split_tensor_to_mixture_parameters(output):
    """ Split up the output nodes into three groups for Pis, Sigmas and Mus to parameterise mixture model. 
    This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850. """
    with tf.name_scope('mixture_split'):
        logits, scales_1, scales_2, locs_1, locs_2 = tf.split(value=output, num_or_size_splits=MDN_SPLITS, axis=1)
        # softmax the mixture weights:
        logits = tf.nn.softmax(logits)
        # Transform the sigmas to e^sigma
        scales_1 = tf.exp(scales_1)
        scales_2 = tf.exp(scales_2)
    return logits, scales_1, scales_2, locs_1, locs_2


def get_mixture_model(logits, locs_1, locs_2, scales_1, scales_2, input_shape):
    with tf.name_scope('mixture_model'):
        cat = Categorical(logits=logits)
        locs = tf.stack([locs_1, locs_2], axis=1)
        scales = tf.stack([scales_1, scales_2], axis=1)
        coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(tf.unstack(locs, axis=-1), tf.unstack(scales, axis=-1))]
        # init_value = tf.zeros(input_shape, dtype=tf.float32)
        mixture = Mixture(cat=cat, components=coll)  # , value=init_value)
    return mixture


def get_loss_func(mixture, Y):
    with tf.name_scope('mixture_loss'):
        loss = mixture.log_prob(Y)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
    return loss


def sample_mixture_model(mixture):
    return mixture.sample()
