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
def midi_input_mapping(default_config):
    """get the default MIDI input mapping."""
    return(default_config["midi"]["input"])

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

def test_midi_message_handling():
    # notes
    input_mapping = [['note_on', 1]]
    note_msg = mido.Message('note_on', channel=0, note=12, velocity=64)
    index, value = utils.midi_message_to_index_value(note_msg, input_mapping)
    assert(index == 0 and value == 12/127)
    # note off
    note_off_msg = mido.Message('note_off', channel=0, note=12, velocity=0)
    try: 
        index, value = utils.midi_message_to_index_value(note_off_msg, input_mapping)
    except ValueError as e:
        # supposed to get a valueerror here.
        pass
    # cc
    input_mapping = [['control_change', 1, 1]]
    cc_msg = mido.Message("control_change", channel=0, control=1, value=64)
    index, value = utils.midi_message_to_index_value(cc_msg, input_mapping)
    assert(index == 0 and value == 64/127)



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
    sender.handle()
    sender.send(output_values)
    sender.disconnect()


def test_serial_midi_server(default_config, sparse_callback, dense_callback, output_values):
    sender = impsio.SerialMIDIServer(
        default_config, sparse_callback, dense_callback
    )
    sender.connect()
    sender.handle()
    sender.send(output_values)
    sender.disconnect()


def test_midi_server(default_config, sparse_callback, dense_callback, output_values):
    sender = impsio.MIDIServer(
        default_config, sparse_callback, dense_callback
    )
    sender.connect()
    sender.handle()
    sender.send(output_values)
    sender.disconnect()
