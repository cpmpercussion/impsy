# Script for Running the RNN Box system
import serial
from serial.tools.list_ports import comports
import time
import struct
import socket
import logging
import datetime
import argparse
from threading import Thread, Condition
import sketch_mdn
import numpy as np
import tensorflow as tf
import queue

# Input and output to serial are bytes (0-255)
# Output to Pd is a float (0-1)
parser = argparse.ArgumentParser(description='Interface for RNN Box.')
parser.add_argument('-l', '--log', dest='logging', action="store_true", help='Save input and RNN data to a log file.')
# Individual Modes
parser.add_argument('-t', '--test', dest='test', action="store_true", help='No RNN, user input only directly connected to servo.')
parser.add_argument('-o', '--only', dest='useronly', action="store_true", help="User control only mode, no RNN or servo.")
parser.add_argument('-r', '--rnn', dest='rnnonly', action="store_true", help='RNN interaction only.')
# Duo Modes
parser.add_argument('-c', '--call', dest='callresponse', action="store_true", help='Call and response mode.')
parser.add_argument('-p', '--polyphony', dest='polyphony', action="store_true", help='Harmony mode.')
parser.add_argument('-b', '--battle', dest='battle', action="store_true", help='Battle royale mode.')
parser.add_argument('-u', '--user', dest='usermodel', action="store_true", help='Use human RNN model.')
parser.add_argument('-s', '--synthetic', dest='syntheticmodel', action='store_true', help='Use synthetic RNN model.')
parser.add_argument('-u', '--untrained', dest='untrainedmodel', action='store_true', help='Use untrained RNN.')

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
    ser = serial.Serial(tty, 115200, timeout=None, write_timeout=None)
except serial.SerialException:
    print("Serial Port busy or not available.")

# Functions for OSC Connection.


def send_sound_command(osc_datagram):
    """Send OSC message via UDP."""
    sock.sendto(osc_datagram, (ADDRESS, PORT))


def pad_dgram_four_bytes(dgram):
    """Pad a datagram up to a multiple of 4 bytes."""
    return (dgram + (b'\x00' * (4 - len(dgram) % 4)))


def touch_message_datagram(address="touch", pos=0.0):
    """Construct an osc message with address /touch and one float."""
    dgram = b''
    dgram += pad_dgram_four_bytes(("/" + address).encode('utf-8'))
    dgram += pad_dgram_four_bytes(b',f')  # (",f"), is this working? test again.
    dgram += struct.pack('>f', pos)
    return dgram

# Functions for sending and receving from levers.


def command_servo(input=128):
    """Send a command to the servo. Input is between 0, 255"""
    ser.write(struct.pack('B', int(input)))


def read_lever():
    """Read a single byte from the lever and return as integer."""
    return ord(ser.read(1))


# ## Load the Model
# Hyperparameters
units = 128
mixes = 8
layers = 1
n_steps = 1
batch_s = 1
sketch_mdn.MODEL_DIR = "./"
# Instantiate Running Network
net = sketch_mdn.MixtureRNN(mode=sketch_mdn.NET_MODE_RUN, n_hidden_units=units, n_mixtures=mixes,
                            batch_size=batch_s, sequence_length=n_steps, layers=layers)
print("RNN Loaded.")
rnn_output_buffer = queue.Queue()
writing_queue = queue.Queue()
# Touch storage for RNN.
last_rnn_touch = sketch_mdn.random_touch()  # prepare previous touch input for RNN input
last_user_touch = sketch_mdn.random_touch()
last_user_interaction = time.time()
CALL_RESPONSE_THRESHOLD = 2.0
call_response_mode = 'call'
# Interaction Loop Parameters
# All set to false before setting is chosen.
thread_running = False
user_to_sound = False
user_to_rnn = False
rnn_to_rnn = False
user_to_servo = False
rnn_to_sound = False
listening_as_well = False


if args.logging:
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=LOG_FORMAT)
    print("Logging enabled:", LOG_FILE)

# Interactive Mapping
if args.test:
    print("Entering test mode (no RNN).")
    # user to sound, user to servo.
    user_to_sound = True
    user_to_servo = True
    user_to_rnn = False
    rnn_to_rnn = False
elif args.callresponse:
    print("Entering call and response mode.")
    # set initial conditions.
    user_to_sound = True
    user_to_rnn = True
    rnn_to_rnn = False
    rnn_to_sound = False
elif args.polyphony:
    print("Entering polyphony mode.")
    user_to_sound = True
    user_to_rnn = True
    rnn_to_rnn = False
    rnn_to_sound = True
elif args.battle:
    print("Entering battle royale mode.")
    user_to_sound = True
    user_to_rnn = False
    rnn_to_rnn = True
    rnn_to_sound = True
elif args.useronly:
    print("Entering user only mode.")
    user_to_sound = True
    user_to_servo = False
elif args.rnnonly:
    print("RNN Playback only mode.")
    user_to_sound = False
    user_to_rnn = False
    rnn_to_rnn = True
    rnn_to_sound = True

# RNN Model choice
if args.usermodel:
    print("Using human RNN model.")
elif args.syntheticmodel:
    print("Using synthetic RNN model.")


def interaction_loop(sess):
    # Interaction loop for the box, reads serial, makes predictions, outputs servo and sound.
    global last_user_touch
    global last_user_interaction
    global last_rnn_touch
    # Start Lever Processing
    userloc = None
    while ser.in_waiting > 0:
        userloc = read_lever()
        userloc = min(max(userloc, 0), 255)
        logging.info("{1},user,{0}".format(userloc, datetime.datetime.now().isoformat()))
        userdt = time.time() - last_user_interaction
        last_user_interaction = time.time()
        last_user_touch = np.array([userdt, userloc / 255.0])
        if user_to_sound:
            send_sound_command(touch_message_datagram(address='user', pos=(userloc / 255.0)))
        if user_to_servo:
            print("Sending User to Servo:", userloc)
            command_servo(userloc)
    # Send any waiting messages to the servo.
    while not writing_queue.empty():
        servo_pos = writing_queue.get()
        command_servo(servo_pos)

    # Make predictions.
    if user_to_rnn and userloc:
        rnn_output = net.generate_touch(last_user_touch, sess)
        print("conditioned RNN state", str(time.time()))
        if rnn_to_sound:
            rnn_output_buffer.put_nowait(rnn_output)
    if rnn_to_rnn and rnn_output_buffer.empty():
        rnn_output = net.generate_touch(last_rnn_touch, sess)
        print("made RNN prediction in:", last_rnn_touch, "out:", rnn_output)
        rnn_output_buffer.put_nowait(rnn_output)  # put it in the playback queue.


def playback_rnn_loop():
    # Plays back RNN notes from its buffer queue.
    global last_rnn_touch
    while True:
        item = rnn_output_buffer.get(block=True, timeout=None)  # Blocks until next item is available.
        print("processing an rnn command", time.time())
        # convert to dt, byte format
        dt = item[0]
        x_loc = min(max(item[1], 0), 1)  # x_loc in [0,1]
        dt = max(dt, 0.001)  # stop accidental minus and zero dt.
        servo_pos = int(255 * x_loc)  # Scale pos to 0-255
        time.sleep(dt)  # wait until time to play the sound
        last_rnn_touch = np.array([dt, x_loc])  # set the last rnn movement to the corrected value.
        if rnn_to_sound:
            # RNN can be disconnected from sound
            send_sound_command(touch_message_datagram(address='rnn', pos=x_loc))
            # command_servo(servo_pos)
            writing_queue.put_nowait(servo_pos)
            print("RNN Played:", servo_pos, "at", dt)
            logging.info("{1},rnn,{0}".format(servo_pos, datetime.datetime.now().isoformat()))
        rnn_output_buffer.task_done()


def monitor_user_action():
    # Handles changing responsibility in Call-Response mode.
    global call_response_mode
    global user_to_rnn
    global rnn_to_rnn
    global rnn_to_sound
    # Check when the last user interaction was
    dt = time.time() - last_user_interaction
    if dt > CALL_RESPONSE_THRESHOLD:
        # switch to response modes.
        user_to_rnn = False
        rnn_to_rnn = True
        rnn_to_sound = True
        if call_response_mode is 'call':
            print("switching to response.")
            call_response_mode = 'response'
    else:
        # switch to call mode.
        user_to_rnn = True
        rnn_to_rnn = False
        rnn_to_sound = False
        if call_response_mode is 'response':
            print("switching to call.")
            call_response_mode = 'call'
            print("Clearning RNN buffer")
            while not rnn_output_buffer.empty():
                rnn_output_buffer.get()
                rnn_output_buffer.task_done()
                print("Cleared an RNN buffer item")
            print("ready for call mode")


print("Now running...")
thread_running = True

with tf.Session() as sess:
    net.prepare_model_for_running(sess)
    # condition = Condition()
    # rnn_thread = Thread(target=playback_rnn_loop, name="rnn_player_thread", args=(condition,), daemon=True)
    rnn_thread = Thread(target=playback_rnn_loop, name="rnn_player_thread", daemon=True)
    try:
        # user_thread.start()
        rnn_thread.start()
        while True:
            interaction_loop(sess)
            if args.callresponse:
                monitor_user_action()
    except KeyboardInterrupt:
        print("\nCtrl-C received... exiting.")
        thread_running = False
        # user_thread.join(timeout=1)
        rnn_thread.join(timeout=1)
        pass
    finally:
        print("\nDone, shutting down.")

# New idea:
# have a flag for "needs a new rnn note" and if that is unset, generate a new note in the interaction loop.
# could have some limit for nuimber to generate, maybe 200ms worth e.g.
# no threaded RNN then, just schedule notes, when they're played from the queue, the flag gets unset.
