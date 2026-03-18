"""TensorFlow version compatibility shim for IMPSY.

Abstracts differences between TF 2.16 (tf-keras, tf.lite) and
TF 2.18+ (native Keras 3) and TF 2.20+ (ai_edge_litert replaces tf.lite).

This module is intended to be thin and temporary — remove once older
TF versions are dropped.
"""

import tensorflow as tf

TF_VERSION = tuple(int(x) for x in tf.__version__.split(".")[:2])

# TF 2.20+ deprecates tf.lite in favour of the ai_edge_litert package.
TF_LITE_DEPRECATED = TF_VERSION >= (2, 20)


def get_tflite_interpreter(model_path: str):
    """Return a TFLite interpreter for inference.

    Prefers tf.lite.Interpreter because it includes the Flex delegate
    needed for models converted with SELECT_TF_OPS (e.g. MDRNN models).
    Falls back to ai_edge_litert if tf.lite is unavailable (TF 2.20+).
    """
    try:
        return tf.lite.Interpreter(model_path=model_path)
    except AttributeError:
        pass
    from ai_edge_litert import interpreter as litert
    return litert.Interpreter(model_path=model_path)


def get_tflite_converter(model):
    """Return a TFLite converter from a Keras model."""
    if TF_LITE_DEPRECATED:
        try:
            from ai_edge_litert import converter as litert_converter
            return litert_converter.TFLiteConverterV2.from_keras_model(model)
        except ImportError:
            pass
    return tf.lite.TFLiteConverter.from_keras_model(model)


def get_tflite_ops():
    """Return (TFLITE_BUILTINS, SELECT_TF_OPS) enum values."""
    if TF_LITE_DEPRECATED:
        try:
            from ai_edge_litert import converter as litert_converter
            return (
                litert_converter.OpsSet.TFLITE_BUILTINS,
                litert_converter.OpsSet.SELECT_TF_OPS,
            )
        except ImportError:
            pass
    return tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS


def get_tflite_optimize_default():
    """Return the default optimisation flag."""
    if TF_LITE_DEPRECATED:
        try:
            from ai_edge_litert import converter as litert_converter
            return litert_converter.Optimize.DEFAULT
        except ImportError:
            pass
    return tf.lite.Optimize.DEFAULT


def analyze_tflite_model(model_content):
    """Run TFLite model analyser if available."""
    if not TF_LITE_DEPRECATED:
        tf.lite.experimental.Analyzer.analyze(model_content=model_content)
