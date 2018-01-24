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
import queue

# Input and output to serial are bytes (0-255)
# Output to Pd is a float (0-1)
parser = argparse.ArgumentParser(description='Interface for RNN Box.')


parser.add_argument('-l', '--log', dest='logging', action="store_true", help='Save input and RNN data to a log file.')
parser.add_argument('-t', '--test', dest='test', action="store_true", help='No RNN, user input only directly connected to servo.')
parser.add_argument('-c', '--call', dest='callresponse', action="store_true", help='Call and response mode.')
parser.add_argument('-p', '--polyphony', dest='polyphony', action="store_true", help='Harmony mode.')
parser.add_argument('-b', '--battle', dest='battle', action="store_true", help='Battle royale mode.')
parser.add_argument('-u', '--user', dest='usermodel', action="store_true", help='Use human RNN model.')
parser.add_argument('-s', '--synthetic', dest='syntheticmodel', action='store_true', help='Use synthetic RNN model.')

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
rnn_output_buffer = queue.Queue()


# Process Args here


# Parameters
# All set to false before setting is chosen.
thread_running = False
user_to_sound = False
user_to_rnn = False
rnn_to_rnn = False
rnn_to_servo = False
user_to_servo = False
rnn_to_sound = False
listening_as_well = False


def random_touch():
    return np.array([(0.01 + (np.random.rand() - 0.5) * 0.005), (np.random.rand() - 0.5)])

# Touch storage for RNN.
last_rnn_touch = random_touch()  # prepare previous touch input for RNN input
last_user_touch = random_touch()
last_user_interaction = time.time()


if args.logging:
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=LOG_FORMAT)
    print("Logging enabled:", LOG_FILE)


# Interactive Mapping
if args.test:
    print("Entering test mode (no RNN).")
    user_to_sound = True
    user_to_servo = False
elif args.callresponse:
    print("Entering call and response mode.")
elif args.polyphony:
    print("Entering polyphony mode.")
elif args.battle:
    print("Entering battle royale mode.")

# TODO make this work.
# RNN Model choice
if args.usermodel:
    print("Using human RNN model.")
elif args.syntheticmodel:
    print("Using synthetic RNN model.")


def rnn_make_prediction(sess):
    input = None
    if user_to_rnn:
        input = last_user_touch
    if rnn_to_rnn:
        input = last_rnn_touch
    if input is None:
        input = random_touch()
    rnn_output = net.generate_touch(input, sess)
    rnn_output_buffer.put_nowait(rnn_output)  # put it in the playback queue.


def playback_user_loop(sess):
    # Plays back serial messages from the user
    global last_user_touch
    global last_user_interaction
    while thread_running:
        loc = None
        while ser.in_waiting > 0:
            loc = read_lever()
            logging.info("{1}, user, {0}".format(loc, datetime.datetime.now().isoformat()))
            send_sound_command(touch_message_datagram(loc / 255.0))
        if user_to_servo and loc:
            command_servo(loc)
        dt = 0.5
        if loc is not None:
            last_user_touch = np.array([dt, loc / 255.0])
            last_user_interaction = time.time()
            if user_to_rnn:
                rnn_make_prediction(sess)


def playback_rnn_loop(sess):
    # Plays back RNN notes from its buffer queue.
    global last_rnn_touch
    while thread_running:
        item = rnn_output_buffer.get()  # could put a timeout here.
        if item is not None:
            # convert to dt, byte format
            dt = item[0]
            pos = item[1]
            pos = int((pos + 10) * 255 / 20.0)  # what does this maths do?
            pos = min(max(pos, 0), 255)  # ditto here?
            dt = max(dt, 0)  # stop accidental minus dt
            time.sleep(dt)  # wait until time to play the sound
            if rnn_to_sound:
                # RNN can be disconnected from sound
                move_and_play_sound(pos)  # do the playing and moving
                logging.info("{1}, rnn, {0}".format(pos, datetime.datetime.now().isoformat()))
            last_rnn_touch = item  # Set the last_rnn_touch (after the time delay?)
            rnn_output_buffer.task_done()
        if rnn_to_rnn:
            rnn_make_prediction(sess)


CALL_RESPONSE_THRESHOLD = 2.0
call_response_mode = 'call'


def monitor_user_action():
    # Handles changing responsibility in Call-Response mode.
    global call_response_mode
    global user_to_rnn
    global rnn_to_rnn
    global rnn_to_sound
    global rnn_to_servo
    while thread_running:
        # Check when the last user interaction was
        dt = time.time() - last_user_interaction
        if dt > CALL_RESPONSE_THRESHOLD and args.callresponse:
            # switch to response modes.
            if call_response_mode is 'call':
                print("switching to response.")
                call_response_mode = 'response'
            user_to_rnn = False
            rnn_to_rnn = True
            rnn_to_sound = True
            rnn_to_servo = True
            rnn_make_prediction(sess)
        elif args.callresponse:
            if call_response_mode is 'response':
                print("switching to call.")
                call_response_mode = 'call'
            user_to_rnn = True
            rnn_to_rnn = False
            rnn_to_sound = False
            rnn_to_servo = False


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
thread_running = True

with tf.Session() as sess:
    net.prepare_model_for_running(sess)
    user_thread = Thread(target=playback_user_loop, args=(sess,), name="user_thread")
    rnn_thread = Thread(target=playback_rnn_loop, args=(sess,), name="rnn_thread")
    try:
        user_thread.start()
        rnn_thread.start()
        while True:
            monitor_user_action()
            # Do nothing in the main thread
            pass
            # playback_user_loop(sess)
    except KeyboardInterrupt:
        thread_running = False
        user_thread.join()
        rnn_thread.join()
        print("threads finished...exiting")
        pass
    finally:
        print("\nDisconnected")

# # Start a thread for listening
# if listening_as_well:
#     thread_running = True
#     thread = Thread(target=threaded_function)
#     thread.start()

## Why aren't any of the loops completing?
## Next idea is to get rid of threading and just do a series of actions, any future sounds can just be scheduled.
