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
from .utils import mdrnn_config
import impsy.impsio as impsio


np.set_printoptions(precision=2)


def match_midi_port_to_list(port, port_list):
    """Return the closest actual MIDI port name given a partial match and a list."""
    if port in port_list:
        return port
    contains_list = [x for x in port_list if port in x]
    if not contains_list:
        return False
    else:
        return contains_list[0]


def setup_logging(dimension, location = "logs/"):
    """Setup a log file and logging, requires a dimension parameter"""
    log_file = datetime.datetime.now().isoformat().replace(":", "-")[:19] + "-" + str(dimension) + "d" +  "-mdrnn.log"  # Log file name.
    log_file = location + log_file
    log_format = '%(message)s'
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format=log_format)
    click.secho(f'Logging enabled: {log_file}', fg='green')


def open_raspberry_serial():
    """Tries to open a serial port for MIDI IO on Raspberry Pi."""
    try:
        click.secho("Trying to open Raspberry Pi serial port for MIDI in/out.", fg='yellow')
        ser = serial.Serial('/dev/ttyAMA0', baudrate=31250)
    except:
        ser = None
        click.secho("Could not open serial port, might be in development mode.", fg='red')
    return ser


def build_network(sess, compute_graph, config):
    """Build the MDRNN, uses a high-level size parameter and dimension."""
    import impsy.mdrnn as mdrnn
    import tensorflow.compat.v1 as tf
    dimension = config["model"]["dimension"]
    size = config["model"]["size"]
    click.secho(f"MDRNN: Using {size} model.", fg="green")
    model_config = mdrnn_config(size)
    mdrnn_units = model_config["units"]
    mdrnn_layers = model_config["layers"]
    mdrnn_mixes = model_config["mixes"]
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


class ImpsySender(object):
    """Class for sending data to external clients via MIDI, WebSockets, Serial, or OSC."""

    def __init__(self, config) -> None:
        self.config = config # Config dictionary read in from config.toml
        self.dimension = self.config["model"]["dimension"] # retrieve dimension from the config file.
        self.last_midi_notes = {} # dict to store last played notes via midi
        # MIDI port opening
        click.secho("Opening MIDI port for input/output.", fg='yellow')
        try:
            desired_input_port = match_midi_port_to_list(self.config["midi"]["in_device"], mido.get_input_names())
            self.midi_in_port = mido.open_input(desired_input_port)
            click.secho(f"MIDI: in port is: {self.midi_in_port.name}", fg='green')
        except: 
            self.midi_in_port = None
            click.secho("Could not open MIDI input.", fg='red')
            click.secho(f"MIDI Inputs: {mido.get_input_names()}", fg = 'blue')
        try:
            desired_output_port = match_midi_port_to_list(self.config["midi"]["out_device"], mido.get_output_names())
            self.midi_out_port = mido.open_output(desired_output_port)
            click.secho(f"MIDI: out port is: {self.midi_out_port.name}", fg='green')
        except:
            self.midi_out_port = None
            click.secho("Could not open MIDI output.", fg='red')
            click.secho(f"MIDI Outputs: {mido.get_output_names()}", fg = 'blue')

    def serial_send_midi(self, message):
        """Sends a mido MIDI message via the very basic serial output on Raspberry Pi GPIO."""
        try:
            self.serial.write(message.bin())
        except: 
            pass

    def send_midi_message(self, message):
        """Send a MIDI message across all required outputs"""
        # TODO: this is where we can have laggy performance, careful.
        if self.midi_out_port is not None:
            self.midi_out_port.send(message)
        self.serial_send_midi(message)
        self.websocket_send_midi(message)

    def send_midi_note_on(self, channel, pitch, velocity):
        """Send a MIDI note on (and implicitly handle note_off)"""
        # stop the previous note
        try:
            midi_msg = mido.Message('note_off', channel=channel, note=self.last_midi_notes[channel], velocity=0)
            self.send_midi_message(midi_msg)
            # click.secho(f"MIDI: note_off: {self.last_midi_notes[channel]}: msg: {midi_msg.bin()}", fg="blue")
            # do this by whatever other channels necessary
        except KeyError:
            click.secho("Something wrong with turning MIDI notes off!!", fg="red")
            pass

        # play the present note
        midi_msg = mido.Message('note_on', channel=channel, note=pitch, velocity=velocity)
        self.send_midi_message(midi_msg)
        # click.secho(f"MIDI: note_on: {pitch}: msg: {midi_msg.bin()}", fg="blue")
        self.last_midi_notes[channel] = pitch

    def send_control_change(self, channel, control, value):
        """Send a MIDI control change message"""
        midi_msg = mido.Message('control_change', channel=channel, control=control, value=value)
        self.send_midi_message(midi_msg)

    def send_midi_note_offs(self):
        """Sends note offs on any MIDI channels that have been used for notes."""
        outconf = self.config["midi"]["output"]
        out_channels = [x[1] for x in outconf if x[0] == "note_on"]
        for i in out_channels:
            try:
                midi_msg = mido.Message('note_off', channel=i-1, note=self.last_midi_notes[i-1], velocity=0)
                self.send_midi_message(midi_msg)
                # click.secho(f"MIDI: note_off: {self.last_midi_notes[i-1]}: msg: {midi_msg.bin()}", fg="blue")
            except KeyError:
                click.secho("Something wrong with all MIDI Note off!", fg="red")
                pass

    def send_sound_command_midi(self, command_args):
        """Sends sound commands via MIDI"""
        assert len(command_args)+1 == self.dimension, "Dimension not same as prediction size." # Todo more useful error.
        start_time = datetime.datetime.now()
        outconf = self.config["midi"]["output"]
        values = list(map(int, (np.ceil(command_args * 127))))
        if VERBOSE:
            click.secho(f'out: {values}', fg='green')

        for i in range(self.dimension-1):
            if outconf[i][0] == "note_on":
                self.send_midi_note_on(outconf[i][1]-1, values[i], 127) # note decremented channel (0-15)
            if outconf[i][0] == "control_change":
                self.send_control_change(outconf[i][1]-1, outconf[i][2], values[i]) # note decrement channel (0-15)
        duration_time = (datetime.datetime.now() - start_time).total_seconds()
        if duration_time > 0.02:
            click.secho(f"Sound command sending took a long time: {(duration_time):.3f}s", fg="red")
        # TODO: is it a good idea to have all this indexing? easy to screw up.

    def handle_midi_input(self):
        """Handle MIDI input messages that might come from mido"""
        if self.midi_in_port is None:
            return # fail early if MIDI not open.
        for message in self.midi_in_port.iter_pending():
            if message.type == "note_on":
                try:
                    index = self.config["midi"]["input"].index(["note_on", message.channel+1])
                    value = message.note / 127.0
                    construct_input_list(index,value)
                except ValueError:
                    pass

            if message.type == "control_change":
                try:
                    index = self.config["midi"]["input"].index(["control_change", message.channel+1, message.control])
                    value = message.value / 127.0
                    construct_input_list(index,value)
                except ValueError:
                    pass


# TODO: some storage for all the output channels.
WS_CLIENTS = set() # storage for potential ws clients.

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


class InteractionServer(object):
    """Interaction server class. Contains state and functions for the interaction loop."""

    def __init__(self):
        """Initialises the interaction server including loading the config from a config.toml file."""
        click.secho("Preparing IMPSY interaction server...", fg="yellow")
        click.secho("Opening configuration.", fg="yellow")
        with open("config.toml", "rb") as f:
            self.config = tomllib.load(f)

        ## Load global variables from the config file.
        self.verbose = self.config["verbose"]
        self.dimension = self.config["model"]["dimension"] # retrieve dimension from the config file.

        self.output_sender = ImpsySender(self.config)


        # Serial port opening
        self.serial = open_raspberry_serial()

        # Import Keras and tensorflow, doing this later to make CLI more responsive.
        click.secho("Importing MDRNN.", fg='yellow')
        start_import = time.time()
        import impsy.mdrnn as mdrnn
        import tensorflow.compat.v1 as tf
        click.secho(f"Done. That took {time.time() - start_import} seconds.", fg='yellow')

        # Interaction Loop Parameters
        # All set to false before setting is chosen.
        self.mode = self.config["interaction"]["mode"]

        # Interactive Mapping
        if self.mode == "callresponse":
            click.secho("Config: call and response mode.", fg='blue')
            self.user_to_rnn = True
            self.rnn_to_rnn = False
            self.rnn_to_sound = False
        elif self.mode == "polyphony":
            click.secho("Config: polyphony mode.", fg='blue')
            self.user_to_rnn = True
            self.rnn_to_rnn = False
            self.rnn_to_sound = True
        elif self.mode == "battle":
            click.secho("Config: battle royale mode.", fg='blue')
            self.user_to_rnn = False
            self.rnn_to_rnn = True
            self.rnn_to_sound = True
        elif self.mode == "useronly":
            click.secho("Config: user only mode.", fg='blue')
            self.user_to_rnn = False
            self.rnn_to_rnn = False
            self.rnn_to_sound = False
        else: 
            click.secho("Config: no mode set, setting to user only")
            self.user_to_rnn = False
            self.rnn_to_rnn = False
            self.rnn_to_sound = False

        # Set up runtime variables.
        self.interface_input_queue = queue.Queue()
        self.rnn_prediction_queue = queue.Queue()
        self.rnn_output_buffer = queue.Queue()
        self.writing_queue = queue.Queue()
        self.last_user_interaction_time = time.time()
        self.last_user_interaction_data = mdrnn.random_sample(out_dim=self.dimension)
        self.rnn_prediction_queue.put_nowait(mdrnn.random_sample(out_dim=self.dimension))
        self.call_response_mode = 'call'


    def construct_input_list(self, index, value):
        """constructs a dense input list from a sparse format (e.g., when receiving MIDI)
        """
        # set up dense interaction list
        int_input = self.last_user_interaction_data[1:]
        int_input[index] = value
        # log
        values = list(map(int, (np.ceil(int_input * 127))))
        if VERBOSE:
            click.secho(f"in: {values}", fg='yellow')
        logging.info("{1},interface,{0}".format(','.join(map(str, int_input)),
                    datetime.datetime.now().isoformat()))
        # put it in the queue
        dt = time.time() - self.last_user_interaction_time
        self.last_user_interaction_time = time.time()
        self.last_user_interaction_data = np.array([dt, *int_input])
        assert len(self.last_user_interaction_data) == self.dimension, "Input is incorrect dimension, set dimension to %r" % len(self.last_user_interaction_data)
        # These values are accessed by the RNN in the interaction loop function.
        self.interface_input_queue.put_nowait(self.last_user_interaction_data)
        # Send values to output if in config
        if self.config["interaction"]["input_thru"]:
                send_sound_command_midi(np.minimum(np.maximum(self.last_user_interaction_data[1:], 0), 1))


    def make_prediction(self, sess, compute_graph, neural_net):
        """Part of the interaction loop: reads input, makes predictions, outputs results"""
        import tensorflow.compat.v1 as tf
        # First deal with user --> MDRNN prediction
        if self.user_to_rnn and not self.interface_input_queue.empty():
            item = self.interface_input_queue.get(block=True, timeout=None)
            tf.keras.backend.set_session(sess)
            with compute_graph.as_default():
                rnn_output = neural_net.generate_touch(item)
            if self.rnn_to_sound:
                self.rnn_output_buffer.put_nowait(rnn_output)
            self.interface_input_queue.task_done()

        # Now deal with MDRNN --> MDRNN prediction.
        if self.rnn_to_rnn and self.rnn_output_buffer.empty() and not self.rnn_prediction_queue.empty():
            item = self.rnn_prediction_queue.get(block=True, timeout=None)
            tf.keras.backend.set_session(sess)
            with compute_graph.as_default():
                rnn_output = neural_net.generate_touch(item)
            self.rnn_output_buffer.put_nowait(rnn_output)  # put it in the playback queue.
            self.rnn_prediction_queue.task_done()


    def monitor_user_action(self):
        """Handles changing responsibility in Call-Response mode."""
        # Check when the last user interaction was
        dt = time.time() - self.last_user_interaction_time
        if dt > self.config["interaction"]["threshold"]:
            # switch to response modes.
            self.user_to_rnn = False
            self.rnn_to_rnn = True
            self.rnn_to_sound = True
            if self.call_response_mode == 'call':
                click.secho("switching to response.", bg='red', fg='black')
                self.call_response_mode = 'response'
                while not self.rnn_prediction_queue.empty():
                    # Make sure there's no inputs waiting to be predicted.
                    self.rnn_prediction_queue.get()
                    self.rnn_prediction_queue.task_done()
                self.rnn_prediction_queue.put_nowait(self.last_user_interaction_data)  # prime the RNN queue
        else:
            # switch to call mode.
            self.user_to_rnn = True
            self.rnn_to_rnn = False
            self.rnn_to_sound = False
            if self.call_response_mode == 'response':
                click.secho("switching to call.", bg='blue', fg='black')
                self.call_response_mode = 'call'
                # Empty the RNN queues.
                while not self.rnn_output_buffer.empty():
                    # Make sure there's no actions waiting to be synthesised.
                    self.rnn_output_buffer.get()
                    self.rnn_output_buffer.task_done()
                # close sound control over MIDI
                send_midi_note_offs()


    def playback_rnn_loop():
        """Plays back RNN notes from its buffer queue. This loop blocks and should run in a separate thread."""
        while True:
            item = self.rnn_output_buffer.get(block=True, timeout=None)  # Blocks until next item is available.
            dt = item[0]
            # click.secho(f"Raw dt: {dt}", fg="blue")
            x_pred = np.minimum(np.maximum(item[1:], 0), 1)
            dt = max(dt, 0.001)  # stop accidental minus and zero dt.
            dt = dt * self.config["model"]["timescale"] # timescale modification!
            # click.secho(f"Sleeping for dt: {dt}", fg="blue")

            time.sleep(dt)  # wait until time to play the sound
            # put last played in queue for prediction.
            self.rnn_prediction_queue.put_nowait(np.concatenate([np.array([dt]), x_pred]))
            if self.rnn_to_sound:
                # send_sound_command(x_pred)
                send_sound_command_midi(x_pred)
                if self.config["log_predictions"]:
                    logging.info("{1},rnn,{0}".format(','.join(map(str, x_pred)),
                                datetime.datetime.now().isoformat()))
            self.rnn_output_buffer.task_done()


    def serve_forever(self):
        """Run the interaction server opening required IO."""
        # Import Keras and tensorflow, doing this later to make CLI more responsive.
        click.secho("Importing MDRNN.", fg='yellow')
        start_import = time.time()
        import impsy.mdrnn as mdrnn
        import tensorflow.compat.v1 as tf
        click.secho(f"Done. That took {time.time() - start_import} seconds.", fg='yellow')

        # Build model
        compute_graph = tf.Graph()
        with compute_graph.as_default():
            sess = tf.Session()
        net = build_network(sess, compute_graph, self.config)

        # Load model weights
        click.secho("Preparing MDRNN.", fg='yellow')
        tf.keras.backend.set_session(sess)
        with compute_graph.as_default():
            if self.config["model"]["file"] != "":
                net.load_model(model_file=self.config["model"]["file"]) # load custom model.
            else:
                net.load_model()  # try loading from default file location.

        # Threads
        click.secho("Preparing MDRNN thread.", fg='yellow')
        rnn_thread = Thread(target=self.playback_rnn_loop, name="rnn_player_thread", daemon=True)
        click.secho("Preparing websocket thread.", fg='yellow')
        ws_thread = Thread(target=websocket_serve_loop, name="ws_receiver_thread", daemon=True)

        # Logging
        if self.config["log"]:
            setup_logging(self.dimension)

        # Start threads and run IO loop
        try:
            rnn_thread.start()
            ws_thread.start()
            click.secho("RNN Thread Started", fg="green")
            while True:
                self.make_prediction(sess, compute_graph, net)
                if self.config["interaction"]["mode"] == "callresponse":
                    self.output_sender.handle_midi_input() # handles incoming midi queue
                    # TODO: handle other kinds of input here?
                    self.monitor_user_action()
        except KeyboardInterrupt:
            click.secho("\nCtrl-C received... exiting.", fg='red')
            rnn_thread.join(timeout=0.1)
            ws_thread.join(timeout=0.1)
            self.output_sender.send_midi_note_offs() # stop all midi notes.
        finally:
            click.secho("\nDone, shutting down.", fg='red')


@click.command(name="start-genai-module")
def start_genai_module():
    """Run IMPSY interaction system with MIDI, WebSockets, and OSC."""
    click.secho("GenAI: Running startup and main loop.", fg="blue")
    interaction_server = InteractionServer()
    interaction_server.serve_forever()
