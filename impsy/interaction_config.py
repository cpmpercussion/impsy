"""impsy.interaction_config: Functions for using imps as an interactive music system. This version uses a config file instead of a CLI."""

import logging
import time
import datetime
import numpy as np
import queue
import serial
import tomllib
from threading import Thread
import mido
import click
from websockets.sync.server import serve


def match_midi_port_to_list(port, port_list):
    """Return the closest actual MIDI port name given a partial match and a list."""
    if port in port_list:
        return port
    contains_list = [x for x in port_list if port in x]
    if not contains_list:
        return False
    else:
        return contains_list[0]
    

click.secho("Opening configuration.", fg="yellow")
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

## Load global variables from the config file.
VERBOSE = config["verbose"]
dimension = config["model"]["dimension"] # retrieve dimension from the config file.

# TODO: some storage for all the output channels.
OUTPUT_CHANNELS = {}
WS_CLIENTS = set() # storage for potential ws clients.

# MIDI port opening
click.secho("Opening MIDI port for input/output.", fg='yellow')
try:
    desired_input_port = match_midi_port_to_list(config["midi"]["in_device"], mido.get_input_names())
    midi_in_port = mido.open_input(desired_input_port)
    click.secho(f"MIDI: in port is: {midi_in_port.name}", fg='green')
except: 
    midi_in_port = None
    click.secho("Could not open MIDI input.", fg='red')
try:
    desired_output_port = match_midi_port_to_list(config["midi"]["out_device"], mido.get_output_names())
    midi_out_port = mido.open_output(desired_output_port)
    click.secho(f"MIDI: out port is: {midi_out_port.name}", fg='green')
except:
    midi_out_port = None
    click.secho("Could not open MIDI output.", fg='red')
    click.secho("Listing MIDI Inputs and Outputs", fg='blue')
    click.secho(f"Input: {mido.get_input_names()}", fg = 'blue')
    click.secho(f"Output: {mido.get_output_names()}", fg = 'blue')

# Serial port opening
try:
    click.secho("Opening Serial Port for MIDI in/out.", fg='yellow')
    ser = serial.Serial('/dev/ttyAMA0', baudrate=31250)
except:
    ser = None
    click.secho("Could not open serial port, might be in development mode.", fg='red')

# Import Keras and tensorflow, doing this later to make CLI more responsive.
click.secho("Importing MDRNN.", fg='yellow')
start_import = time.time()
import impsy.mdrnn as mdrnn
import tensorflow.compat.v1 as tf
click.secho(f"Done. That took {time.time() - start_import} seconds.", fg='yellow')

# Interaction Loop Parameters
# All set to false before setting is chosen.
user_to_rnn = False
rnn_to_rnn = False
rnn_to_sound = False

# Interactive Mapping
if config["interaction"]["mode"] == "callresponse":
    click.secho("Config: call and response mode.", fg='blue')
    # set initial conditions.
    user_to_rnn = True
    rnn_to_rnn = False
    rnn_to_sound = False
elif config["interaction"]["mode"] == "polyphony":
    click.secho("Config: polyphony mode.", fg='blue')
    user_to_rnn = True
    rnn_to_rnn = False
    rnn_to_sound = True
elif config["interaction"]["mode"] == "battle":
    click.secho("Config: battle royale mode.", fg='blue')
    user_to_rnn = False
    rnn_to_rnn = True
    rnn_to_sound = True
elif config["interaction"]["mode"] == "useronly":
    click.secho("Config: user only mode.", fg='blue')
    user_to_rnn = False
    rnn_to_rnn = False
    rnn_to_sound = False


def build_network(sess, compute_graph, size, dimension):
    """Build the MDRNN, uses a high-level size parameter and dimension."""
    # Choose model parameters.
    if size == 'xs':
        click.secho("MDRNN: Using XS model.", fg="green")
        mdrnn_units = 32
        mdrnn_mixes = 5
        mdrnn_layers = 2
    elif size == 's':
        click.secho("MDRNN: Using S model.", fg="green")
        mdrnn_units = 64
        mdrnn_mixes = 5
        mdrnn_layers = 2
    elif size == 'm':
        click.secho("MDRNN: Using M model.", fg="green")
        mdrnn_units = 128
        mdrnn_mixes = 5
        mdrnn_layers = 2
    elif size == 'l':
        click.secho("MDRNN: Using L model.", fg="green")
        mdrnn_units = 256
        mdrnn_mixes = 5
        mdrnn_layers = 2
    elif size == 'xl':
        click.secho("MDRNN: Using XL model.", fg="green")
        mdrnn_units = 512
        mdrnn_mixes = 5
        mdrnn_layers = 3
    # construct the model
    mdrnn.MODEL_DIR = "./models/"
    tf.keras.backend.set_session(sess)
    with compute_graph.as_default():
        net = mdrnn.PredictiveMusicMDRNN(mode=mdrnn.NET_MODE_RUN,
                                              dimension=dimension,
                                              n_hidden_units=mdrnn_units,
                                              n_mixtures=mdrnn_mixes,
                                              layers=mdrnn_layers)
        net.pi_temp = config["model"]["pitemp"]
        net.sigma_temp = config["model"]["sigmatemp"]
    click.secho(f"MDRNN Loaded: {net.model_name()}", fg="green")
    return net


def make_prediction(sess, compute_graph, neural_net):
    """Part of the interaction loop: reads input, makes predictions, outputs results"""

    # First deal with user --> MDRNN prediction
    if user_to_rnn and not interface_input_queue.empty():
        item = interface_input_queue.get(block=True, timeout=None)
        tf.keras.backend.set_session(sess)
        with compute_graph.as_default():
            rnn_output = neural_net.generate_touch(item)
        if rnn_to_sound:
            rnn_output_buffer.put_nowait(rnn_output)
        interface_input_queue.task_done()

    # Now deal with MDRNN --> MDRNN prediction.
    if rnn_to_rnn and rnn_output_buffer.empty() and not rnn_prediction_queue.empty():
        item = rnn_prediction_queue.get(block=True, timeout=None)
        tf.keras.backend.set_session(sess)
        with compute_graph.as_default():
            rnn_output = neural_net.generate_touch(item)
        rnn_output_buffer.put_nowait(rnn_output)  # put it in the playback queue.
        rnn_prediction_queue.task_done()


def send_sound_command_midi(command_args):
    """Sends sound commands via MIDI"""
    assert len(command_args)+1 == dimension, "Dimension not same as prediction size." # Todo more useful error.
    start_time = datetime.datetime.now()
    outconf = config["midi"]["output"]
    values = list(map(int, (np.ceil(command_args * 127))))
    if VERBOSE:
        click.secho(f'out: {values}', fg='green')

    for i in range(dimension-1):
        if outconf[i][0] == "note_on":
            send_midi_note_on(outconf[i][1]-1, values[i], 127) # note decremented channel (0-15)
        if outconf[i][0] == "control_change":
            send_control_change(outconf[i][1]-1, outconf[i][2], values[i]) # note decrement channel (0-15)
    duration_time = (datetime.datetime.now() - start_time).total_seconds()
    if duration_time > 0.02:
        click.secho(f"Sound command sending took a long time: {(duration_time):.3f}s", fg="red")
    # TODO: is it a good idea to have all this indexing? easy to screw up.


last_midi_notes = {} # dict to store last played notes via midi

def send_midi_note_on(channel, pitch, velocity):
    """Send a MIDI note on (and implicitly handle note_off)"""
    global last_midi_notes
    # stop the previous note
    try:
        midi_msg = mido.Message('note_off', channel=channel, note=last_midi_notes[channel], velocity=0)
        send_midi_message(midi_msg)
        # click.secho(f"MIDI: note_off: {last_midi_notes[channel]}: msg: {midi_msg.bin()}", fg="blue")
        # do this by whatever other channels necessary
    except KeyError:
        click.secho("Something wrong with turning MIDI notes off!!", fg="red")
        pass

    # play the present note
    midi_msg = mido.Message('note_on', channel=channel, note=pitch, velocity=velocity)
    send_midi_message(midi_msg)
    # click.secho(f"MIDI: note_on: {pitch}: msg: {midi_msg.bin()}", fg="blue")
    last_midi_notes[channel] = pitch


def send_midi_note_offs():
    """Sends note offs on any MIDI channels that have been used for notes."""
    global last_midi_notes
    outconf = config["midi"]["output"]
    out_channels = [x[1] for x in outconf if x[0] == "note_on"]
    for i in out_channels:
        try:
            midi_msg = mido.Message('note_off', channel=i-1, note=last_midi_notes[i-1], velocity=0)
            send_midi_message(midi_msg)
            # click.secho(f"MIDI: note_off: {last_midi_notes[i-1]}: msg: {midi_msg.bin()}", fg="blue")
        except KeyError:
            click.secho("Something wrong with all MIDI Note off!", fg="red")
            pass


def send_control_change(channel, control, value):
    """Send a MIDI control change message"""
    midi_msg = mido.Message('control_change', channel=channel, control=control, value=value)
    send_midi_message(midi_msg)


def send_midi_message(msg):
    """Send a MIDI message across all required outputs"""
    # TODO: this is where we can have laggy performance, careful.
    midi_out_port.send(msg)
    serial_send_midi(msg)
    websocket_send_midi(msg)


def serial_send_midi(message):
    """Sends a mido MIDI message via the very basic serial output on Raspberry Pi GPIO."""
    try:
        ser.write(message.bin())
    except: 
        pass


def playback_rnn_loop():
    """Plays back RNN notes from its buffer queue. This loop blocks and should run in a separate thread."""
    while True:
        item = rnn_output_buffer.get(block=True, timeout=None)  # Blocks until next item is available.
        dt = item[0]
        # click.secho(f"Raw dt: {dt}", fg="blue")
        x_pred = np.minimum(np.maximum(item[1:], 0), 1)
        dt = max(dt, 0.001)  # stop accidental minus and zero dt.
        dt = dt * config["model"]["timescale"] # timescale modification!
        # click.secho(f"Sleeping for dt: {dt}", fg="blue")

        time.sleep(dt)  # wait until time to play the sound
        # put last played in queue for prediction.
        rnn_prediction_queue.put_nowait(np.concatenate([np.array([dt]), x_pred]))
        if rnn_to_sound:
            # send_sound_command(x_pred)
            send_sound_command_midi(x_pred)
            if config["log_predictions"]:
                logging.info("{1},rnn,{0}".format(','.join(map(str, x_pred)),
                            datetime.datetime.now().isoformat()))
        rnn_output_buffer.task_done()


def construct_input_list(index, value):
    """constructs a dense input list from a sparse format (e.g., when receiving MIDI)
    """
    global last_user_interaction_time
    global last_user_interaction_data
    # set up dense interaction list
    int_input = last_user_interaction_data[1:]
    int_input[index] = value
    # log
    values = list(map(int, (np.ceil(int_input * 127))))
    if VERBOSE:
        click.secho(f"in: {values}", fg='yellow')
    logging.info("{1},interface,{0}".format(','.join(map(str, int_input)),
                 datetime.datetime.now().isoformat()))
    # put it in the queue
    dt = time.time() - last_user_interaction_time
    last_user_interaction_time = time.time()
    last_user_interaction_data = np.array([dt, *int_input])
    assert len(last_user_interaction_data) == dimension, "Input is incorrect dimension, set dimension to %r" % len(last_user_interaction_data)
    # These values are accessed by the RNN in the interaction loop function.
    interface_input_queue.put_nowait(last_user_interaction_data)
    # Send values to output if in config
    if config["interaction"]["input_thru"]:
            send_sound_command_midi(np.minimum(np.maximum(last_user_interaction_data[1:], 0), 1))


def handle_midi_input():
    """Handle MIDI input messages that might come from mido"""
    if midi_in_port is None:
        return # fail early if MIDI not open.
    for message in midi_in_port.iter_pending():
        if message.type == "note_on":
            try:
                index = config["midi"]["input"].index(["note_on", message.channel+1])
                value = message.note / 127.0
                construct_input_list(index,value)
            except ValueError:
                pass

        if message.type == "control_change":
            try:
                index = config["midi"]["input"].index(["control_change", message.channel+1, message.control])
                value = message.value / 127.0
                construct_input_list(index,value)
            except ValueError:
                pass


def websocket_send_midi(message):
    """Sends a mido MIDI message via websockets if available."""
    global ws_client

    if message.type == "note_on":
        ws_msg = f"/channel/{message.channel}/noteon/{message.note}/{message.velocity}"
    elif message.type == "note_off":
        ws_msg = f"/channel/{message.channel}/noteoff/{message.note}/{message.velocity}"
    elif message.type == "control_change":
        ws_msg = f"/channel/{message.channel}/cc/{message.control}/{message.value}"
    else:
        return
    # click.secho(f"WS out: {ws_msg}")
    # Broadcast the ws_msg to all clients (sync version can't use websockets.broadcast function so doing this naively)
    for ws_client in WS_CLIENTS.copy():
        try:
            ws_client.send(ws_msg)
        except:
            WS_CLIENTS.remove(ws_client)


def websocket_handler(websocket):
    """Handle websocket input messages that might arrive"""
    global WS_CLIENTS
    WS_CLIENTS.add(websocket) # add websocket to the client list.
    # do the actual handling
    for message in websocket:
        click.secho(f"WS: {message}", fg="red") # TODO: fine for debug, but should be removed really.
        m = message.split('/')[1:]
        msg_type = m[2]
        chan = int(m[1]) # TODO: should this be chan+1 or -1 or something.
        note = int(m[3])
        vel = int(m[4])
        if msg_type == "noteon":
            # note_on
            try:
                index = config["midi"]["input"].index(["note_on", chan])
                value = note / 127.0
                construct_input_list(index,value)
            except ValueError:
                click.secho(f"WS in: exception with message {message}", fg="red")
                pass
        elif msg_type == "cc":
            # cc
            try:
                index = config["midi"]["input"].index(["control_change", chan, note])
                value = vel / 127.0
                construct_input_list(index,value)
            except ValueError:
                click.secho(f"WS in: exception with message {message}", fg="red")
                pass
        # global websocket
        # ws_msg = f"/channel/{message.channel}/noteon/{message.note}/{message.velocity}"
        # ws_msg = f"/channel/{message.channel}/noteoff/{message.note}/{message.velocity}"
        # ws_msg = f"/channel/{message.channel}/cc/{message.control}/{message.value}"


def websocket_serve_loop():
    """Threading websockets server following https://websockets.readthedocs.io/en/stable/reference/sync/server.html"""
    hostname = config['websocket']['server_ip']
    port = config['websocket']['server_port']
    with serve(websocket_handler, hostname, port) as server:
        server.serve_forever()


def monitor_user_action():
    """Handles changing responsibility in Call-Response mode."""
    global call_response_mode
    global user_to_rnn
    global rnn_to_rnn
    global rnn_to_sound
    # Check when the last user interaction was
    dt = time.time() - last_user_interaction_time
    if dt > config["interaction"]["threshold"]:
        # switch to response modes.
        user_to_rnn = False
        rnn_to_rnn = True
        rnn_to_sound = True
        if call_response_mode == 'call':
            click.secho("switching to response.", bg='red', fg='black')
            call_response_mode = 'response'
            while not rnn_prediction_queue.empty():
                # Make sure there's no inputs waiting to be predicted.
                rnn_prediction_queue.get()
                rnn_prediction_queue.task_done()
            rnn_prediction_queue.put_nowait(last_user_interaction_data)  # prime the RNN queue
    else:
        # switch to call mode.
        user_to_rnn = True
        rnn_to_rnn = False
        rnn_to_sound = False
        if call_response_mode == 'response':
            click.secho("switching to call.", bg='blue', fg='black')
            call_response_mode = 'call'
            # Empty the RNN queues.
            while not rnn_output_buffer.empty():
                # Make sure there's no actions waiting to be synthesised.
                rnn_output_buffer.get()
                rnn_output_buffer.task_done()
            # close sound control over MIDI
            send_midi_note_offs()


def setup_logging(dimension, location = "logs/"):
    """Setup a log file and logging, requires a dimension parameter"""
    log_file = datetime.datetime.now().isoformat().replace(":", "-")[:19] + "-" + str(dimension) + "d" +  "-mdrnn.log"  # Log file name.
    log_file = location + log_file
    log_format = '%(message)s'
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format=log_format)
    click.secho(f'Logging enabled: {log_file}', fg='green')


# Set up runtime variables.
interface_input_queue = queue.Queue()
rnn_prediction_queue = queue.Queue()
rnn_output_buffer = queue.Queue()
writing_queue = queue.Queue()
last_user_interaction_time = time.time()
last_user_interaction_data = mdrnn.random_sample(out_dim=dimension)
rnn_prediction_queue.put_nowait(mdrnn.random_sample(out_dim=dimension))
call_response_mode = 'call'

@click.command(name="start-genai-module")
def start_genai_module():
    """Run IMPSY interaction system with MIDI, WebSockets, and OSC."""
    click.secho("GenAI: Running startup and main loop.", fg="blue")
    # Build model
    compute_graph = tf.Graph()
    with compute_graph.as_default():
        sess = tf.Session()
    net = build_network(sess, compute_graph, config["model"]["size"], config["model"]["dimension"])

    # Load model weights
    click.secho("Preparing MDRNN.", fg='yellow')
    tf.keras.backend.set_session(sess)
    with compute_graph.as_default():
        if config["model"]["file"] != "":
            net.load_model(model_file=config["model"]["file"]) # load custom model.
        else:
            net.load_model()  # try loading from default file location.

    # Threads
    click.secho("Preparing MDRNN thread.", fg='yellow')
    rnn_thread = Thread(target=playback_rnn_loop, name="rnn_player_thread", daemon=True)
    click.secho("Preparing websocket thread.", fg='yellow')
    ws_thread = Thread(target=websocket_serve_loop, name="ws_receiver_thread", daemon=True)

    # Logging
    if config["log"]:
        setup_logging(dimension)

    # Start threads and run IO loop
    try:
        rnn_thread.start()
        ws_thread.start()
        click.secho("RNN Thread Started", fg="green")
        while True:
            make_prediction(sess, compute_graph, net)
            if config["interaction"]["mode"] == "callresponse":
                handle_midi_input() # handles incoming midi queue
                # TODO: handle other kinds of input here?
                monitor_user_action()
    except KeyboardInterrupt:
        click.secho("\nCtrl-C received... exiting.", fg='red')
        rnn_thread.join(timeout=0.1)
        ws_thread.join(timeout=0.1)
        send_midi_note_offs() # stop all midi notes.
    finally:
        click.secho("\nDone, shutting down.", fg='red')

