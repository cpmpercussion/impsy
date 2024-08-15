from impsy import impsio, utils
import pytest
import numpy as np
import time
from pathlib import Path
import mido


@pytest.fixture(scope="session")
def default_config():
    """get the default config file."""
    config_path = Path("configs") / "default.toml"
    config = utils.get_config_data(config_path)
    return(config)

@pytest.fixture(scope="session")
def midi_output_mapping(default_config):
    """get the default MIDI output mapping."""
    return(default_config["midi"]["output"])


@pytest.fixture(scope="session")
def output_values(default_config):
    dimension = default_config["model"]["dimension"]
    return np.random.rand(dimension - 1)

@pytest.fixture(scope="session")
def last_midi_notes_dict(midi_output_mapping):
    """produce a dict of previous played midi notes"""
    last_midi_notes = {}
    out_channels = [x[1] for x in midi_output_mapping if x[0] == "note_on"] # just get channels associated with note_on messages.
    for chan in out_channels:
        last_midi_notes[chan] = 60 # played middle c on each output channel.
    return last_midi_notes



@pytest.fixture(scope="session")
def sparse_callback():
    def callback():
        return
    return callback


@pytest.fixture(scope="session")
def dense_callback():
    def callback():
        return 
    return callback


# test utils.


def test_midi_mapping_to_output(output_values, midi_output_mapping):
    output_messages = utils.output_values_to_midi_messages(output_values, midi_output_mapping)
    assert len(output_messages) == len(output_values), "Number of output messages does not match number of output values"
    for msg in output_messages:
        assert isinstance(msg, mido.Message), "msg is not a mido.Message object"


def test_midi_note_off_generation(midi_output_mapping, last_midi_notes_dict):
    output_messages = utils.get_midi_note_offs(midi_output_mapping, last_midi_notes_dict)
    for msg in output_messages:
        assert isinstance(msg, mido.Message), "msg is not a mido.Message object"
        assert msg.type == "note_off", "msg is not a note_off"    


# test IOServers


def test_websocket_server(default_config, sparse_callback, dense_callback, output_values):
    sender = impsio.WebSocketServer(
        default_config, sparse_callback, dense_callback
    )
    sender.connect()
    sender.handle()
    sender.send(output_values)
    time.sleep(0.1)
    sender.disconnect()
    time.sleep(0.1)
    # test if we can open the socket that was just closed.



def test_osc_server(default_config, sparse_callback, dense_callback, output_values):
    sender = impsio.OSCServer(
        default_config, sparse_callback, dense_callback
    )
    sender.connect()
    sender.handle()
    sender.send(output_values)
    time.sleep(0.1)
    sender.disconnect()
    time.sleep(0.1)


def test_serial_server(default_config, sparse_callback, dense_callback, output_values):
    sender = impsio.SerialServer(
        default_config, sparse_callback, dense_callback
    )
    sender.connect()
    # sender.handle()
    sender.send(output_values)
    sender.disconnect()


# def test_serial_midi_server(default_config, sparse_callback, dense_callback, output_values):
#     sender = impsio.SerialMIDIServer(
#         default_config, sparse_callback, dense_callback
#     )
#     sender.connect()
#     # sender.handle()
#     sender.send(output_values)
#     sender.disconnect()


def test_midi_server(default_config, sparse_callback, dense_callback, output_values):
    sender = impsio.MIDIServer(
        default_config, sparse_callback, dense_callback
    )
    sender.connect()
    sender.handle()
    sender.send(output_values)
    sender.disconnect()
