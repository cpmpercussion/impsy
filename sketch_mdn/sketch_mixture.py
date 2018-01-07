""" Mixture Model Operations as used in Sketch RNN """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

N_MIXTURES = 24
MDN_SPLITS = 6 # (pi, sigma_1, sigma_2, mu_1, mu_2, rho)
N_OUTPUT_UNITS = N_MIXTURES * MDN_SPLITS

def split_tensor_to_mixture_parameters(output):
    """ Split up the output nodes into three groups for Pis, Sigmas and Mus to parameterise mixture model. 
    This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850. """
    with tf.name_scope('mixture_split'):
        logits, scales_1, scales_2, locs_1, locs_2, corr = tf.split(value=output, num_or_size_splits=MDN_SPLITS, axis=1)
        # softmax the mixture weights:
        pis = tf.nn.softmax(logits)
        # Transform the sigmas to e^sigma
        scales_1 = tf.exp(scales_1)
        scales_2 = tf.exp(scales_2)
        # Transform the correlations to tanh(corr)
        corr = tf.tanh(corr)
    return pis, scales_1, scales_2, locs_1, locs_2, corr
       
def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """ Returns the  probability of (x1,x2) occuring in the bivariate
    gaussian model parameterised by mu1, mu2, s1, s2, rho. 
    Following eq # 24 and 25 of http://arxiv.org/abs/1308.0850."""
    with tf.name_scope('2d_normal_prob'):
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
             2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        neg_rho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * neg_rho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
        result = tf.div(result, denom)
    return result

def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
    """Returns a loss function for a mixture of bivariate normal distributions given a true value.
    Based on eq #26 of http://arxiv.org/abs/1308.0850."""
    with tf.name_scope('mixture_loss'):
        result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        epsilon = 1e-6
        result1 = tf.multiply(result0, z_pi)
        result1 = tf.reduce_sum(result1, 1, keep_dims=True)
        result1 = -tf.log(result1 + epsilon)  # avoid log(0)
    return result1

def adjust_temp(pi_pdf, temp):
    """ Adjusts temperature of a PDF describing a categorical model """
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a categorical model PDF, optionally greedily."""
    if greedy:
        return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    tf.logging.info('Error sampling mixture model.')
    return -1

def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def sample_mixture_model(pi, mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    """ Takes a sample from a mixture of bivariate normals, with temporature and greediness. """
    idx = get_pi_idx(random.random(), pi, temp, greedy)
    x1, x2 = sample_gaussian_2d(mu1[idx], mu2[idx], s1[idx], s2[idx], rho[idx], np.sqrt(temp), greedy)
    return x1, x2
