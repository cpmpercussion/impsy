"""
Tests to start working with a tflite interpreter.
"""

import numpy as np
import tensorflow as tf
from impsy import mdrnn
# import keras_mdn_layer as mdn
# import time
# import datetime
from pathlib import Path

MODEL = Path("models") / "musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=str(MODEL))
interpreter.allocate_tensors()
# Get the list of signatures
signatures = interpreter.get_signature_list()
print("Signatures:\n", signatures)

# Get the signature runner
runner = interpreter.get_signature_runner()
hidden_units = 64

states = []
for i in range(2):
    states += [
        np.zeros((1, hidden_units), dtype=np.float32),
        np.zeros((1, hidden_units), dtype=np.float32),
    ]

value = np.array(mdrnn.random_sample(out_dim=9), dtype=np.float32)
print(value)
# for i in range(num_test_steps):
#     value = net.generate_touch(value)
#     proc_touch = mdrnn.proc_generated_touch(value, dimension)

# input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

output = runner(
    inputs = value,
    state_h_0 = states[0],
    state_c_0 = states[1],
    state_h_1 = states[2],
    state_c_1 = states[3],
)

print(output)



def reset_lstm_states(self):

    assert (
        len(states) == self.n_rnn_layers * 2
    ), "length of states list needs to be RNN layers times 2 (h and c for each)"
    self.lstm_states = states

# # Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# print("Input details")
# print(input_details)
# print("Output details:")
# print(output_details)

# Prepare input data
# Replace this with your actual input data
# input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

# Set the input tensor
# interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
# interpreter.invoke()

# Get the output tensor
# output_data = interpreter.get_tensor(output_details[0]['index'])

# print(output_data)
