"""impsy.interaction_config: Functions for using imps as an interactive music system. This version uses a config file instead of a CLI."""

import logging
import time
import datetime
import numpy as np
import queue
import tomllib
from threading import Thread
import click
from .utils import mdrnn_config
import impsy.impsio as impsio

np.set_printoptions(precision=2)


def setup_logging(dimension, location = "logs/"):
    """Setup a log file and logging, requires a dimension parameter"""
    log_file = datetime.datetime.now().isoformat().replace(":", "-")[:19] + "-" + str(dimension) + "d" +  "-mdrnn.log"  # Log file name.
    log_file = location + log_file
    log_format = '%(message)s'
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format=log_format)
    click.secho(f'Logging enabled: {log_file}', fg='green')


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
        self.mode = self.config["interaction"]["mode"]

        ## Set up IO.
        self.midi_sender = impsio.MIDIServer(self.config, self.construct_input_list)
        self.midi_sender.connect()
        self.websocket_sender = impsio.WebSocketServer(self.config, self.construct_input_list)

        # Import Keras and tensorflow
        ## TODO may be better to do this only when needed.
        click.secho("Importing MDRNN.", fg='yellow')
        start_import = time.time()
        import impsy.mdrnn as mdrnn
        import tensorflow.compat.v1 as tf
        click.secho(f"Done. That took {time.time() - start_import} seconds.", fg='yellow')

        # Interaction Loop Parameters
        # All set to false before setting is chosen.

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
        self.last_user_interaction_time = time.time()
        self.last_user_interaction_data = mdrnn.random_sample(out_dim=self.dimension)
        self.rnn_prediction_queue.put_nowait(mdrnn.random_sample(out_dim=self.dimension))
        self.call_response_mode = 'call'


    def send_back_values(self, output_values):
        """sends back sound commands to the MIDI/OSC/WebSockets outputs"""
        output = np.minimum(np.maximum(output_values, 0), 1)
        self.midi_sender.send(output)
        self.websocket_sender.send(output)


    # Todo this is the "callback" for our IO functions.
    def construct_input_list(self, index: int, value: float) -> None:
        """constructs a dense input list from a sparse format (e.g., when receiving MIDI)
        """
        # set up dense interaction list
        int_input = self.last_user_interaction_data[1:]
        int_input[index] = value
        # log
        values = list(map(int, (np.ceil(int_input * 127))))
        if self.verbose:
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
                # This is where outputs are sent via impsio objects.
                output_values = np.minimum(np.maximum(self.last_user_interaction_data[1:], 0), 1)
                self.send_back_values(output_values)

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
                self.midi_sender.disconnect()


    def playback_rnn_loop(self):
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
                # Send predictions to outputs via impsio objects
                self.send_back_values(x_pred)
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
        # self.websocket_sender.connect() # TODO verify websockets again.

        # Logging
        if self.config["log"]:
            setup_logging(self.dimension)

        # Start threads and run IO loop
        try:
            rnn_thread.start()
            click.secho("RNN Thread Started", fg="green")
            while True:
                self.make_prediction(sess, compute_graph, net)
                if self.config["interaction"]["mode"] == "callresponse":
                    self.midi_sender.handle_midi_input() # handles incoming midi queue
                    # TODO: handle other kinds of input here?
                    self.monitor_user_action()
        except KeyboardInterrupt:
            click.secho("\nCtrl-C received... exiting.", fg='red')
            rnn_thread.join(timeout=0.1)
            
            self.midi_sender.disconnect() # send note offs and disconnect
            self.websocket_sender.disconnect()
        finally:
            click.secho("\nDone, shutting down.", fg='red')


@click.command(name="start-genai-module")
def start_genai_module():
    """Run IMPSY interaction system with MIDI, WebSockets, and OSC."""
    click.secho("GenAI: Running startup and main loop.", fg="blue")
    interaction_server = InteractionServer()
    interaction_server.serve_forever()
