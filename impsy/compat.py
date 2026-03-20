"""TensorFlow version compatibility shim for IMPSY.

Abstracts differences between TF 2.16 (tf-keras, tf.lite) and
TF 2.18+ (native Keras 3, tf.lite).

This module is intended to be thin and temporary — remove once older
TF versions are dropped.
"""

import tensorflow as tf


def get_tflite_interpreter(model_path: str):
    """Return a TFLite interpreter for inference."""
    return tf.lite.Interpreter(model_path=model_path)


def get_tflite_converter(model):
    """Return a TFLite converter from a Keras model.

    Uses from_concrete_functions to support pure Keras 3 layers
    (from_keras_model fails with Keras 3 custom layers on TF 2.16-2.18).
    """
    input_spec = [tf.TensorSpec(shape=inp.shape, dtype=tf.float32) for inp in model.inputs]

    @tf.function(input_signature=input_spec)
    def serve(*args):
        return model(list(args))

    concrete = serve.get_concrete_function()
    return tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)


def get_tflite_ops():
    """Return (TFLITE_BUILTINS, SELECT_TF_OPS) enum values."""
    return tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS


def get_tflite_optimize_default():
    """Return the default optimisation flag."""
    return tf.lite.Optimize.DEFAULT


def analyze_tflite_model(model_content):
    """Run TFLite model analyser if available."""
    tf.lite.experimental.Analyzer.analyze(model_content=model_content)
