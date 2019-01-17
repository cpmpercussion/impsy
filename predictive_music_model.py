import logging
import time
import datetime
import numpy as np
import queue
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client
import argparse
from threading import Thread

# Input and output to serial are bytes (0-255)
# Output to Pd is a float (0-1)
parser = argparse.ArgumentParser(description='Predictive Musical Interaction MDRNN Interface.')
parser.add_argument('-l', '--log', dest='logging', action="store_true", help='Save input and RNN data to a log file.')
parser.add_argument('-g', '--nogui', dest='nogui', action='store_true', help='Disable the TKinter GUI.')
# Individual Modes
parser.add_argument('-o', '--only', dest='useronly', action="store_true", help="User control only mode, no RNN.")
parser.add_argument('-r', '--rnn', dest='rnnonly', action="store_true", help='RNN interaction only.')
# Duo Modes
parser.add_argument('-c', '--call', dest='callresponse', action="store_true", help='Call and response mode.')
parser.add_argument('-p', '--polyphony', dest='polyphony', action="store_true", help='Harmony mode.')
parser.add_argument('-b', '--battle', dest='battle', action="store_true", help='Battle royale mode.')
# OSC addresses
parser.add_argument("--clientip", default="localhost", help="The address of output device.")
parser.add_argument("--clientport", type=int, default=5000, help="The port the output device is listening on.")
parser.add_argument("--serverip", default="localhost", help="The address of this server.")
parser.add_argument("--serverport", type=int, default=5001, help="The port this server should listen on.")
# MDRNN arguments.
parser.add_argument('-d', '--dimension', type=int, dest='dimension', default=2, help='The dimension of the data to model, must be >= 2.')
parser.add_argument("--modelsize", default="s", help="The model size: s, m, l, xl")
parser.add_argument("--sigmatemp", type=float, default=0.01, help="The sigma temperature for sampling.")
parser.add_argument("--pitemp", type=float, default=1, help="The pi temperature for sampling.")
args = parser.parse_args()

# Import Keras and tensorflow, doing this later to make CLI more responsive.
print("Importing Keras and MDRNN.")
start_import = time.time()
import empi_mdrnn
import tensorflow as tf
from keras import backend as K
print("Done. That took", time.time() - start_import, "seconds.")

# TODO: set up OSC server
# TODO: set up OSC client
# TODO set up interface to build MDRNN
# TODO set up run loop for inference.


# Choose model parameters.
if args.modelsize is 's':
    mdrnn_units = 64
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize is 'm':
    mdrnn_units = 128
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize is 'l':
    mdrnn_units = 256
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize is 'xl':
    mdrnn_units = 512
    mdrnn_mixes = 5
    mdrnn_layers = 3
else:
    mdrnn_units = 128
    mdrnn_mixes = 5
    mdrnn_layers = 2

# Interaction Loop Parameters
# All set to false before setting is chosen.
user_to_rnn = False
rnn_to_rnn = False
rnn_to_sound = False
listening_as_well = False

# Interactive Mapping
if args.callresponse:
    print("Entering call and response mode.")
    # set initial conditions.
    user_to_rnn = True
    rnn_to_rnn = False
    rnn_to_sound = False
elif args.polyphony:
    print("Entering polyphony mode.")
    user_to_rnn = True
    rnn_to_rnn = False
    rnn_to_sound = True
elif args.battle:
    print("Entering battle royale mode.")
    user_to_rnn = False
    rnn_to_rnn = True
    rnn_to_sound = True
elif args.useronly:
    print("Entering user only mode.")
elif args.rnnonly:
    print("RNN Playback only mode.")
    user_to_rnn = False
    rnn_to_rnn = True
    rnn_to_sound = True


def build_network(sess):
    """Build the MDRNN."""
    empi_mdrnn.MODEL_DIR = "./models/"
    K.set_session(sess)
    with compute_graph.as_default():
        net = empi_mdrnn.PredictiveMusicMDRNN(mode=empi_mdrnn.NET_MODE_RUN,
                                              dimension=args.dimension,
                                              n_hidden_units=mdrnn_units,
                                              n_mixtures=mdrnn_mixes,
                                              layers=mdrnn_layers)
        net.pi_temp = args.pitemp
        net.sigma_temp = args.sigmatemp
    print("MDRNN Loaded.")
    return net


def handle_interface_message(address: str, *osc_arguments) -> None:
    """Handler for OSC messages from the interface"""
    global last_user_touch
    global last_user_interaction
    global last_rnn_touch
    int_input = osc_arguments
    logging.info("{1},interface,{0}".format(','.join(map(str, int_input)),
                 datetime.datetime.now().isoformat()))
    dt = time.time() - last_user_interaction
    last_user_interaction = time.time()
    last_user_touch = np.array([dt, *int_input])
    assert len(last_user_touch) == args.dimension, "Input is incorrect dimension, set dimension to %r" % len(last_user_touch)
    # These values are accessed by the RNN in the interaction loop function.
    interface_input_queue.put_nowait(last_user_touch)


def request_rnn_prediction(input_value):
    """ Accesses a single prediction from the RNN. """
    start = time.time()
    output_value = net.generate_touch(input_value)
    time_delta = time.time() - start
    #print("Prediction took:", time_delta)
    return output_value


def make_prediction(sess, compute_graph):
    # Interaction loop: reads input, makes predictions, outputs results.
    # Make predictions.

    # First deal with user --> MDRNN prediction
    if user_to_rnn and not interface_input_queue.empty:
        item = interface_input_queue.get(block=True, timeout=None)
        K.set_session(sess)
        with compute_graph.as_default():
            rnn_output = request_rnn_prediction(item)
        print("conditioned RNN state", str(time.time()))
        if rnn_to_sound:
            rnn_output_buffer.put_nowait(rnn_output)
        interface_input_queue.task_done()

    # Now deal with MDRNN --> MDRNN prediction.
    if rnn_to_rnn and rnn_output_buffer.empty():
        K.set_session(sess)
        with compute_graph.as_default():
            rnn_output = request_rnn_prediction(last_rnn_touch)
        print("made RNN prediction in:", last_rnn_touch, "out:", rnn_output)
        rnn_output_buffer.put_nowait(rnn_output)  # put it in the playback queue.


def send_sound_command(command_args):
    """Send a sound command back to the interface/synth"""
    assert len(command_args)+1 == args.dimension, "Dimension not same as prediction size." # Todo more useful error.
    osc_client.send_message(OUTPUT_MESSAGE_ADDRESS, command_args)


def playback_rnn_loop():
    # Plays back RNN notes from its buffer queue.
    global last_rnn_touch
    while True:
        item = rnn_output_buffer.get(block=True, timeout=None)  # Blocks until next item is available.
        print("processing an rnn command", time.time())
        # convert to dt, byte format
        dt = item[0]
        x_pred = np.minimum(np.maximum(item[1:], 0), 1)
        dt = max(dt, 0.001)  # stop accidental minus and zero dt.
        time.sleep(dt)  # wait until time to play the sound
        last_rnn_touch = np.concatenate([np.array([dt]), x_pred])
        if rnn_to_sound:
            send_sound_command(x_pred)
            print("RNN Played:", x_pred, "at", dt)
            logging.info("{1},rnn,{0}".format(x_pred, datetime.datetime.now().isoformat()))
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

# Logging
LOG_FILE = datetime.datetime.now().isoformat().replace(":", "-")[:19] + "-" + str(args.dimension) + "d" +  "-mdrnn.log"  # Log file name.
LOG_FILE = "logs/" + LOG_FILE
LOG_FORMAT = '%(message)s'

if args.logging:
    logging.basicConfig(filename=LOG_FILE,
                        level=logging.INFO,
                        format=LOG_FORMAT)
    print("Logging enabled:", LOG_FILE)
# Details for OSC output
INPUT_MESSAGE_ADDRESS = "/interface"
OUTPUT_MESSAGE_ADDRESS = "/prediction"

# Set up runtime variables.
# ## Load the Model
compute_graph = tf.Graph()
with compute_graph.as_default():
    sess = tf.Session()
net = build_network(sess)
interface_input_queue = queue.Queue()
rnn_output_buffer = queue.Queue()
writing_queue = queue.Queue()
# Touch storage for RNN.
last_rnn_touch = empi_mdrnn.random_sample(out_dim=args.dimension)
last_user_touch = empi_mdrnn.random_sample(out_dim=args.dimension)
last_user_interaction = time.time()
CALL_RESPONSE_THRESHOLD = 2.0
call_response_mode = 'call'

# Set up OSC client and server
osc_client = udp_client.SimpleUDPClient(args.clientip, args.clientport)
disp = dispatcher.Dispatcher()
disp.map(INPUT_MESSAGE_ADDRESS, handle_interface_message)
server = osc_server.ThreadingOSCUDPServer((args.serverip, args.serverport), disp)

thread_running = True  # todo is this line needed?

# Set up run loop.
print("Preparing MDRNN.")
K.set_session(sess)
with compute_graph.as_default():
    net.load_model()  # try loading from default file location.
print("Preparting MDRNN thread.")
rnn_thread = Thread(target=playback_rnn_loop, name="rnn_player_thread", daemon=True)
print("Preparing Server thread.")
server_thread = Thread(target=server.serve_forever, name="server_thread", daemon=True)

try:
    rnn_thread.start()
    server_thread.start()
    print("Prediction server started.")
    print("Serving on {}".format(server.server_address))
    while True:
        make_prediction(sess, compute_graph)
        if args.callresponse:
            monitor_user_action()
except KeyboardInterrupt:
    print("\nCtrl-C received... exiting.")
    thread_running = False
    rnn_thread.join(timeout=1)
    server_thread.join(timeout=1)
    pass
finally:
    print("\nDone, shutting down.")
