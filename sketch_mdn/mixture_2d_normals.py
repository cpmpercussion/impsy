""" Mixture Model Operations as used in SketchRNN: A Neural Representation of Sketch Drawings. David Ha, Douglas Eck. 2017. http://arxiv.org/abs/1704.03477
Updated by Charles P. Martin for musical MDN experiments. 2018.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

MDN_SPLITS = 6  # (pi, sigma_1, sigma_2, mu_1, mu_2, rho)


def split_tensor_to_mixture_parameters(output):
    """ Split up the output nodes into three groups for Pis, Sigmas and Mus to parameterise mixture model.
    This uses eqns 20, 21, 22 of http://arxiv.org/abs/1308.0850. """
    with tf.name_scope('2d_mixture_split'):
        logits, scales_1, scales_2, locs_1, locs_2, corr = tf.split(value=output, num_or_size_splits=MDN_SPLITS, axis=1, name="2d_params")
        # softmax the mixture weights:
        pis = tf.nn.softmax(logits)
        pis = tf.clip_by_value(pis, 1e-8, 1., name="logits_2d")  # clip pis at 0.
        # Transform the sigmas to e^sigma (ish, using ELU now.)
        scales_1 = tf.add(tf.nn.elu(scales_1), 1. + 1e-8, name="scales_2d_1")  # shape and clip scales
        scales_2 = tf.add(tf.nn.elu(scales_2), 1. + 1e-8, name="scales_2d_2")  # shape and clip scales
        locs_1 = tf.identity(locs_1, name="locs_2d_1")
        locs_2 = tf.identity(locs_2, name="locs_2d_2")
        # Transform the correlations to tanh(corr)
        # TODO: Perhaps clip the corr to 1e-8 as well.
        corr = tf.tanh(corr, name="corr_2d")  # shape corr.
    return pis, scales_1, scales_2, locs_1, locs_2, corr


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """ Returns the  probability of (x1,x2) occuring in the bivariate
    gaussian model parameterised by mu1, mu2, s1, s2, rho.
    Following eq # 24 and 25 of http://arxiv.org/abs/1308.0850.
    This version of the PDF has some offsets in the denominators to account for divide by zero errors.
    """
    with tf.name_scope('2d_normal_prob'):
        epsilon = 1e-8  # protect against divide by zero.
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = (tf.square(tf.divide(norm1, s1)) + tf.square(tf.divide(norm2, s2)) -
             2. * tf.divide(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        one_neg_rho_sq = 1 - tf.square(rho)
        two_one_neg_rho_sq = 2. * one_neg_rho_sq + epsilon  # possible div zero
        result_rhs = tf.exp(tf.divide(-z, two_one_neg_rho_sq))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(one_neg_rho_sq)) + epsilon  # possible div zero
        normal_x_usp = tf.divide(result_rhs, denom)
    return normal_x_usp


def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
    """Returns a loss function for a mixture of bivariate normal distributions given a true value.
    Based on the left-hand side of eq #26 of http://arxiv.org/abs/1308.0850."""
    with tf.name_scope('2d_mixture_loss'):
        summed_probs = tf_2d_mixture_prob(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data)
        epsilon = 1e-6
        neg_log_prob = tf.negative(tf.log(summed_probs + epsilon))  # avoid log(0)
    return neg_log_prob


def tf_2d_mixture_prob(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
    """ Returns the 'probability' of (x1,x2) occurring in the mixture of 2D normals.
    Based on the left-hand side of eq #26 of http://arxiv.org/abs/1308.0850. """
    with tf.name_scope('2d_mixture_prob'):
        kernel_probs = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        weighted_probs = tf.multiply(kernel_probs, z_pi)
        summed_probs = tf.reduce_sum(weighted_probs, 1, keep_dims=True)
        # tf.summary.histogram("1d_kernel_probs", kernel_probs)
        # tf.summary.histogram("1d_weighted_probs", weighted_probs)
        # tf.summary.histogram("1d_summed_probs", summed_probs)
        # tf.summary.histogram("1d_kernel_weights", z_pi)
    return summed_probs


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


def sample_categorical(dist, temp=1.0):
    """Samples a categorical distribution with optional temp adjustment."""
    pdf = adjust_temp(np.copy(dist), temp)
    sample = np.random.multinomial(1, pdf)
    for idx, val in np.ndenumerate(sample):
        if val == 1:
            return idx[0]
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
    # idx = get_pi_idx(random.random(), pi, temp, greedy)
    idx = sample_categorical(pi, temp)
    x1, x2 = sample_gaussian_2d(mu1[idx], mu2[idx], s1[idx], s2[idx], rho[idx], np.sqrt(temp), greedy)
    return x1, x2
