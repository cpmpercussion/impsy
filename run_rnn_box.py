# Script for Running the RNN Box system
import serial
from serial.tools.list_ports import comports
import time
import struct
import socket
import logging
import datetime
import argparse
from threading import Thread
import sketch_mdn
import numpy as np
import tensorflow as tf

# Input and output to serial are bytes (0-255)
# Output to Pd is a float (0-1)

parser = argparse.ArgumentParser(description='Runs the RNN interaction box interface, use the command line to change modes.')
parser.add_argument('-l', '--log', dest='logging', action="store_true", help='Save input and RNN data to a log file.')
parser.add_argument('-t', '--test', dest='test', action="store_true", help='No RNN, user input only directly connected to servo.')
parser.add_argument('-c', '--call', dest='callresponse', action="store_true", help='Call and response mode.')
parser.add_argument('-p', '--polyphony', dest='polyphone', action="store_true", help='Harmony mode.')
parser.add_argument('-b', '--battle', dest='battle', action="store_true", help='Battle royale mode.')
args = parser.parse_args()

LOG_FILE = datetime.datetime.now().isoformat().replace(":", "-")[:19] + "-rnnbox.log"  # Log file name.
LOG_FORMAT = '%(message)s'

# ## OSC and Serial Communication
# Details for OSC output
ADDRESS = "localhost"
PORT = 5000
INT_FLOAT_DGRAM_LEN = 4
STRING_DGRAM_PAD = 4
# Socket for OSC output.
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(0)


def detect_arduino_tty():
    """ Attempts to detect a Myo Bluetooth adapter among the system's serial ports. """
    for p in comports():
        if p[1] == 'SparkFun Pro Micro':
            return p[0]
    return None

# Serial for input and output via a USB-connected microcontroller.
try:
    tty = detect_arduino_tty()
    print("Connecting to", tty)
    ser = serial.Serial(tty, 9600)
except serial.SerialException:
    print("Serial Port busy or not available.")

# Functions for OSC Connection.


def send_sound_command(osc_datagram):
    """Send OSC message via UDP."""
    sock.sendto(osc_datagram, (ADDRESS, PORT))


def pad_dgram_four_bytes(dgram):
    """Pad a datagram up to a multiple of 4 bytes."""
    return (dgram + (b'\x00' * (4 - len(dgram) % 4)))


def touch_message_datagram(pos=0.0):
    """Construct an osc message with address /touch and one float."""
    dgram = b''
    dgram += pad_dgram_four_bytes("/touch".encode('utf-8'))
    dgram += pad_dgram_four_bytes(b',f')  # (",f"), is this working? test again.
    dgram += struct.pack('>f', pos)
    return dgram

# Functions for sending and receving from levers.


def command_servo(input=128):
    """Send a command to the servo. Input is between 0, 255"""
    ser.write(struct.pack('B', input))


def read_lever():
    """Read a single byte from the lever and return as integer."""
    return ord(ser.read(1))

# Functions for playing back level sounds and moving servo


def move_and_play_sound(loc):
    """Move the servo and play a sound in response to a byte input."""
    command_servo(loc)
    send_sound_command(touch_message_datagram(loc / 255.0))


def clear_and_play_serial_input(movement=True):
    """Clears the serial buffer of input and plays it all."""
    while ser.in_waiting > 0:
        loc = read_lever()
        send_sound_command(touch_message_datagram(loc / 255.0))
        if movement:
            command_servo(loc)


# ## Load the Model
#
# Loads a `musical_mdn` model with one predicted variable and time.
sketch_mdn.MODEL_DIR = "./"
# Alternative Network
# net = musical_mdn.TinyJamNet2D(mode = musical_mdn.NET_MODE_RUN, n_hidden_units = 128, n_mixtures = 10, batch_size = 1, sequence_length = 1)
# Main network
net = sketch_mdn.MixtureRNN(mode=sketch_mdn.NET_MODE_RUN, n_hidden_units=128, n_mixtures=10, batch_size=1, sequence_length=1)
print("RNN Loaded.")


# Process Args here

if args.logging:
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=LOG_FORMAT)
    print("Logging enabled:", LOG_FILE)

# Parameters
thread_running = True
user_to_sound = True
user_to_rnn = False
rnn_to_rnn = False
rnn_to_servo = False
user_to_servo = True
rnn_to_sound = False

listening_as_well = False
first_touch = np.array([(0.01 + (np.random.rand() - 0.5) * 0.005), (np.random.rand() - 0.5)])


def threaded_function():
    # Reacts to serial with sound, not movement
    while thread_running:
        while ser.in_waiting > 0:
            loc = read_lever()
            logging.info("{1}, user, {0}".format(loc, datetime.datetime.now().isoformat()))
            send_sound_command(touch_message_datagram(loc / 255.0))
        if user_to_servo:
            command_servo(loc)

# # Start a thread for listening
# if listening_as_well:
#     thread_running = True
#     thread = Thread(target=threaded_function)
#     thread.start()

# # Start TF Predictions
# with tf.Session() as sess:
#     # Get network ready to run
#     net.prepare_model_for_running(sess)
#     last_touch = first_touch
#     time_total = 0
#     count = 0

#     print("Performing a 30s performance")
#     while time_total < 30.0:
#         # generate some output and schedule sounds and movement.
#         last_touch = net.generate_touch(last_touch, sess)
#         output = last_touch.reshape((2,))
#         # convert to dt, byte format
#         dt = output[0]
#         pos = output[1]
#         pos = int((pos + 10) * 255 / 20.0)
#         pos = min(max(pos, 0), 255)
#         dt = max(dt, 0)
#         time.sleep(dt)
#         move_and_play_sound(pos)
#         time_total += dt
#         count += 1
#     print("Done, time was", time_total, "with", count, "moves.")
#     keep_reacting_to_lever = False

# if listening_as_well:
#     thread.join()
#     print("thread finished...exiting")



print("Now running...")
try:
    while True:
        threaded_function()
except KeyboardInterrupt:
    pass
finally:
    m.disconnect()
    print("\nDisconnected")
