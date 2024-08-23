"""
Tests to start working with a tflite interpreter.
"""

import numpy as np
import tensorflow as tf
from impsy import mdrnn
import keras_mdn_layer as mdn
import time
from pathlib import Path
import click

np.set_printoptions(precision=2)

MODEL = Path("models") / "musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
SCALE_FACTOR = 10  # scales input and output from the model. Should be the same between training and inference.

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=str(MODEL))
# interpreter.allocate_tensors()
# Get the list of signatures
signatures = interpreter.get_signature_list()
print("Signatures:\n", signatures)

input_details = interpreter.get_input_details()
# print(input_details)

# Get the signature runner
runner = interpreter.get_signature_runner()
hidden_units = 64

def make_prediction(prev_value, states, runner):
    """makes a prediction using the given runner. Needs to know the exact state names at the moment."""
    input_value = prev_value.reshape(1,1,9) * SCALE_FACTOR
    input_value = input_value.astype(np.float32, copy=False)
    raw_out = runner(
        inputs = input_value,
        state_h_0 = states[0],
        state_c_0 = states[1],
        state_h_1 = states[2],
        state_c_1 = states[3],
    )
    new_states = [raw_out['lstm_0'], raw_out['lstm_0_1'], raw_out['lstm_1'], raw_out['lstm_1_1']]
    # sample from the MDN:
    mdn_params = raw_out['mdn_outputs'].squeeze()
    new_sample = (
        mdn.sample_from_output(
            mdn_params,
            9, # dimension
            5, # num mixtures
            temp=1.5,
            sigma_temp=0.01,
        )
        / SCALE_FACTOR
    )
    new_sample = new_sample.reshape(9,)
    return new_sample, new_states

# create starting values.
states = []
for i in range(2):
    states += [
        np.zeros((1, hidden_units), dtype=np.float32),
        np.zeros((1, hidden_units), dtype=np.float32),
    ]

value = np.array(mdrnn.random_sample(out_dim=9), dtype=np.float32)

tests = 10000
start_time = time.time()
for i in range(tests):
    value, states = make_prediction(value, states, runner)
    value = mdrnn.proc_generated_touch(value, 9)
    click.secho(f"Value: {value}",fg="green")
time_used = time.time() - start_time
time_per_pred = time_used * 1000 / tests
click.secho(
    f"Done in {round(time_used, 2)}s, that's {round(time_per_pred, 2)}ms per prediction!",
    fg="yellow",
)

# [
#     {'name': 'serving_default_state_h_1:0', 'index': 0, 'shape': array([ 1, 64], dtype=int32), 'shape_signature': array([-1, 64], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, 
#     {'name': 'serving_default_state_c_1:0', 'index': 1, 'shape': array([ 1, 64], dtype=int32), 'shape_signature': array([-1, 64], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, 
#     {'name': 'serving_default_inputs:0', 'index': 2, 'shape': array([1, 1, 9], dtype=int32), 'shape_signature': array([-1,  1,  9], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, 
#     {'name': 'serving_default_state_c_0:0', 'index': 3, 'shape': array([ 1, 64], dtype=int32), 'shape_signature': array([-1, 64], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, 
#     {'name': 'serving_default_state_h_0:0', 'index': 4, 'shape': array([ 1, 64], dtype=int32), 'shape_signature': array([-1, 64], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}
# ]
