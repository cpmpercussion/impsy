"""impsy.interaction: Functions for using imps as an interactive music system. This server has OSC input and output."""

import logging
import time
import datetime
import numpy as np
import queue
import click
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
from threading import Thread
from .utils import mdrnn_config


np.set_printoptions(precision=2)

# Interaction Modes:

# user, callresponse, filter, battle.
# user: no AI generation, but user interaction is passed on and logged.
# callresponse: AI responds after 'threshold' seconds, typical turn-based arrangement.
# filter: AI responds directly to every input, could be providing second part etc.
# battle: AI disconnected from human input, both running simultaneously.


# Global variables
# TODO: get rid of these, use a Class instead for data storage.
call_response_mode = False
user_to_rnn = False
rnn_to_rnn = False
rnn_to_sound = False
last_user_interaction_time = None
last_user_interaction_data = None


@click.command(name="run")
@click.option(
    "--log/--no-log", default=True, help="Save input and output data to a log file."
)
@click.option(
    "--verbose/--no-verbose",
    default=True,
    help="Verbose mode, print prediction results.",
)
# Performance modes
@click.option(
    "-O",
    "--mode",
    type=str,
    default="callresponse",
    help="Select interaction mode, one of: user, callresponse, filter, battle. user: no AI generation, callresponse: AI responds after 'threshold' seconds, filter: AI responds directly to every input, battle: AI disconnected from human input",
)
@click.option(
    "-T",
    "--threshold",
    type=float,
    default=2.0,
    help="Seconds to wait before switching to response",
)
# MDRNN arguments.
@click.option(
    "-D",
    "--dimension",
    type=int,
    default=2,
    help="The dimension of the data to model, must be >= 2.",
)
@click.option(
    "-M", "--modelsize", default="s", help="The model size: xs, s, m, l, xl", type=str
)
@click.option(
    "-S",
    "--sigmatemp",
    type=float,
    default=0.01,
    help="The sigma temperature for sampling.",
)
@click.option(
    "-P", "--pitemp", type=float, default=1, help="The pi temperature for sampling."
)
# OSC addresses
@click.option(
    "--clientip",
    type=str,
    default="localhost",
    help="The address of output device, default is 'localhost'.",
)
@click.option(
    "--clientport",
    type=int,
    default=5000,
    help="The port the output device is listening on, default is 5000.",
)
@click.option(
    "--serverip",
    type=str,
    default="localhost",
    help="The address of this server, default is 'localhost'.",
)
@click.option(
    "--serverport",
    type=int,
    default=5001,
    help="The port this server should listen on, default is 5001.",
)
def run(
    log: bool,
    verbose: bool,
    mode: str,
    threshold: float,
    dimension: int,
    modelsize: str,
    sigmatemp: float,
    pitemp: float,
    clientip: str,
    clientport: int,
    serverip: str,
    serverport: int,
):
    """Run IMPS predictive musical interaction system as a server with OSC input and output."""
    global call_response_mode
    global user_to_rnn
    global rnn_to_rnn
    global rnn_to_sound
    global last_user_interaction_time
    global last_user_interaction_data

    # import tensorflow, do this now to make CLI more responsive.
    print("Importing MDRNN.")
    start_import = time.time()
    import impsy.mdrnn.mdrnn as mdrnn
    import tensorflow.compat.v1 as tf

    print("Done. That took", time.time() - start_import, "seconds.")

    model_config = mdrnn_config(modelsize)
    mdrnn_units = model_config["units"]
    mdrnn_layers = model_config["layers"]
    mdrnn_mixes = model_config["mixes"]

    # Interaction Loop Parameters
    # All set to false before setting is chosen.
    user_to_rnn = False
    rnn_to_rnn = False
    rnn_to_sound = False

    # Interactive Mapping
    if mode == "callresponse":
        print("Entering call and response mode.")
        # set initial conditions.
        user_to_rnn = True
        rnn_to_rnn = False
        rnn_to_sound = False
    elif mode == "filter":
        print("Entering filter mode.")
        user_to_rnn = True
        rnn_to_rnn = False
        rnn_to_sound = True
    elif mode == "battle":
        print("Entering battle royale mode.")
        user_to_rnn = False
        rnn_to_rnn = True
        rnn_to_sound = True
    elif mode == "user":
        print("Entering user only mode.")
        user_to_rnn = False
        rnn_to_rnn = False
        rnn_to_sound = False

    def build_network(sess):
        """Build the MDRNN."""
        mdrnn.MODEL_DIR = "./models/"
        tf.keras.backend.set_session(sess)
        with compute_graph.as_default():
            net = mdrnn.PredictiveMusicMDRNN(
                mode=mdrnn.NET_MODE_RUN,
                dimension=dimension,
                n_hidden_units=mdrnn_units,
                n_mixtures=mdrnn_mixes,
                layers=mdrnn_layers,
            )
            net.pi_temp = pitemp
            net.sigma_temp = sigmatemp
        print("MDRNN Loaded.")
        return net

    def handle_interface_message(address: str, *osc_arguments) -> None:
        """Handler for OSC messages from the interface"""
        global last_user_interaction_time
        global last_user_interaction_data
        if verbose:
            # print out OSC message.
            print(
                "User:", ", ".join(["{0:0.2f}".format(abs(i)) for i in osc_arguments])
            )
        int_input = osc_arguments
        logger = logging.getLogger("impslogger")
        logger.info(
            "{1},interface,{0}".format(
                ",".join(map(str, int_input)), datetime.datetime.now().isoformat()
            )
        )
        dt = time.time() - last_user_interaction_time
        last_user_interaction_time = time.time()
        last_user_interaction_data = np.array([dt, *int_input])
        assert (
            len(last_user_interaction_data) == dimension
        ), "Input is incorrect dimension, set dimension to %r" % len(
            last_user_interaction_data
        )
        # These values are accessed by the RNN in the interaction loop function.
        interface_input_queue.put_nowait(last_user_interaction_data)

    def handle_temperature_message(address: str, *osc_arguments) -> None:
        """Handler for temperature messages from the interface: format is ff [sigma temp, pi temp]"""
        new_sigma_temp = osc_arguments[0]
        new_pi_temp = osc_arguments[1]
        if verbose:
            print(f"Temperature -- Sigma: {new_sigma_temp}, Pi: {new_pi_temp}")
        net.sigma_temp = new_sigma_temp
        net.pi_temp = new_pi_temp

    def handle_timescale_message(address: str, *osc_arguments) -> None:
        """Handler for timescale messages: format is f [timescale]"""
        new_timescale = osc_arguments[0]
        if verbose:
            print(f"Timescale: {new_timescale}")
        # TODO: implement this on the prediction end.

    def request_rnn_prediction(input_value):
        """Accesses a single prediction from the RNN."""
        output_value = net.generate_touch(input_value)
        return output_value

    def make_prediction(sess, compute_graph):
        """Interaction loop: reads input, makes predictions, outputs results."""
        # Make predictions.

        # First deal with user --> MDRNN prediction
        if user_to_rnn and not interface_input_queue.empty():
            item = interface_input_queue.get(block=True, timeout=None)
            tf.keras.backend.set_session(sess)
            with compute_graph.as_default():
                rnn_output = request_rnn_prediction(item)
            # if verbose:
            #     print("User->RNN:", ",".join(["{0:0.2f}".format(float(i)) for i in rnn_output]))
            if rnn_to_sound:
                rnn_output_buffer.put_nowait(rnn_output)
            interface_input_queue.task_done()

        # Now deal with MDRNN --> MDRNN prediction.
        if (
            rnn_to_rnn
            and rnn_output_buffer.empty()
            and not rnn_prediction_queue.empty()
        ):
            item = rnn_prediction_queue.get(block=True, timeout=None)
            tf.keras.backend.set_session(sess)
            with compute_graph.as_default():
                rnn_output = request_rnn_prediction(item)
            if verbose:
                print(
                    "RNN: ",
                    ", ".join(["{0:0.2f}".format(abs(i)) for i in rnn_output[1:]]),
                    "dt:",
                    "{0:0.2f}".format(abs(rnn_output[0])),
                )
            rnn_output_buffer.put_nowait(rnn_output)  # put it in the playback queue.
            rnn_prediction_queue.task_done()

    def send_sound_command(command_args):
        """Send a sound command back to the interface/synth"""
        assert (
            len(command_args) + 1 == dimension
        ), "Dimension not same as prediction size."  # Todo more useful error.
        osc_client.send_message(OUTPUT_MESSAGE_ADDRESS, command_args)

    def playback_rnn_loop():
        """Plays back RNN notes from its buffer queue."""
        while True:
            item = rnn_output_buffer.get(
                block=True, timeout=None
            )  # Blocks until next item is available.
            # print("processing an rnn command", time.time())
            dt = item[0]
            x_pred = np.minimum(np.maximum(item[1:], 0), 1)
            dt = max(dt, 0.001)  # stop accidental minus and zero dt.
            time.sleep(dt)  # wait until time to play the sound
            # put last played in queue for prediction.
            rnn_prediction_queue.put_nowait(np.concatenate([np.array([dt]), x_pred]))
            if rnn_to_sound:
                send_sound_command(x_pred)
                # print("RNN Played:", x_pred, "at", dt)
                logger = logging.getLogger("impslogger")
                logger.info(
                    "{1},rnn,{0}".format(
                        ",".join(map(str, x_pred)), datetime.datetime.now().isoformat()
                    )
                )
            rnn_output_buffer.task_done()

    def monitor_user_action():
        """Handles changing action responsibility in Call-Response mode."""
        global call_response_mode
        global user_to_rnn
        global rnn_to_rnn
        global rnn_to_sound
        # Check when the last user interaction was
        dt = time.time() - last_user_interaction_time
        if dt > threshold:
            # switch to response modes.
            user_to_rnn = False
            rnn_to_rnn = True
            rnn_to_sound = True
            if call_response_mode == "call":
                print("switching to response.")
                call_response_mode = "response"
                while not rnn_prediction_queue.empty():
                    # Make sure there's no inputs waiting to be predicted.
                    rnn_prediction_queue.get()
                    rnn_prediction_queue.task_done()
                rnn_prediction_queue.put_nowait(
                    last_user_interaction_data
                )  # prime the RNN queue
        else:
            # switch to call mode.
            user_to_rnn = True
            rnn_to_rnn = False
            rnn_to_sound = False
            if call_response_mode == "response":
                print("switching to call.")
                call_response_mode = "call"
                # Empty the RNN queues.
                while not rnn_output_buffer.empty():
                    # Make sure there's no actions waiting to be synthesised.
                    rnn_output_buffer.get()
                    rnn_output_buffer.task_done()

    # Logging
    LOG_FILE = (
        datetime.datetime.now().isoformat().replace(":", "-")[:19]
        + "-"
        + str(dimension)
        + "d"
        + "-mdrnn.log"
    )  # Log file name.
    LOG_FILE = "logs/" + LOG_FILE
    LOG_FORMAT = "%(message)s"

    if log:
        formatter = logging.Formatter(LOG_FORMAT)
        handler = logging.FileHandler(LOG_FILE)
        handler.setFormatter(formatter)
        logger = logging.getLogger("impslogger")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        print("Logging enabled:", LOG_FILE)
    # Details for OSC output
    INPUT_MESSAGE_ADDRESS = "/interface"
    OUTPUT_MESSAGE_ADDRESS = "/prediction"
    TEMPERATURE_MESSAGE_ADDRESS = "/temperature"
    TIMESCALE_MESSAGE_ADDRESS = "/timescale"

    # Set up runtime variables.
    # ## Load the Model
    compute_graph = tf.Graph()
    with compute_graph.as_default():
        sess = tf.Session()
    net = build_network(sess)
    interface_input_queue = queue.Queue()
    rnn_prediction_queue = queue.Queue()
    rnn_output_buffer = queue.Queue()
    writing_queue = queue.Queue()
    last_user_interaction_time = time.time()
    last_user_interaction_data = mdrnn.random_sample(out_dim=dimension)
    rnn_prediction_queue.put_nowait(mdrnn.random_sample(out_dim=dimension))
    call_response_mode = "call"

    # Set up OSC client and server
    osc_client = udp_client.SimpleUDPClient(clientip, clientport)
    disp = dispatcher.Dispatcher()
    disp.map(INPUT_MESSAGE_ADDRESS, handle_interface_message)
    disp.map(TEMPERATURE_MESSAGE_ADDRESS, handle_temperature_message)
    disp.map(TIMESCALE_MESSAGE_ADDRESS, handle_timescale_message)
    server = osc_server.ThreadingOSCUDPServer((serverip, serverport), disp)

    thread_running = True  # TODO: is this line needed?

    # Set up run loop.
    print("Preparing MDRNN.")
    tf.keras.backend.set_session(sess)
    with compute_graph.as_default():
        net.load_model()  # try loading from default file location.
    print("Preparting MDRNN thread.")
    rnn_thread = Thread(target=playback_rnn_loop, name="rnn_player_thread", daemon=True)
    print("Preparing Server thread.")
    server_thread = Thread(
        target=server.serve_forever, name="server_thread", daemon=True
    )

    try:
        rnn_thread.start()
        server_thread.start()
        print("Prediction server started.")
        print("Serving on {}".format(server.server_address))
        while True:
            make_prediction(sess, compute_graph)
            if mode == "callresponse":
                monitor_user_action()
    except KeyboardInterrupt:
        print("\nCtrl-C received... exiting.")
        thread_running = False
        rnn_thread.join(timeout=0.1)
        server_thread.join(timeout=0.1)
        pass
    finally:
        print("\nDone, shutting down.")
