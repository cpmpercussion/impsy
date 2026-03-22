"""TensorFlow version compatibility shim for IMPSY.

Abstracts differences between TF 2.16 (tf-keras, tf.lite) and
TF 2.18+ (native Keras 3, tf.lite).

This module is intended to be thin and temporary — remove once older
TF versions are dropped.
"""

import tensorflow as tf


def get_tflite_interpreter(model_path: str):
    """Return a TFLite interpreter for inference."""
    from ai_edge_litert.interpreter import Interpreter
    return Interpreter(model_path=model_path)


def get_tflite_optimize_default():
    """Return the default optimisation flag."""
    return tf.lite.Optimize.DEFAULT


def analyze_tflite_model(model_content):
    """Run TFLite model analyser if available."""
    tf.lite.experimental.Analyzer.analyze(model_content=model_content)
