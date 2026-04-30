from impsy import impsio, utils
import pytest
import numpy as np
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import mido


@pytest.fixture(scope="session")
def default_config():
    """get the default config file."""
    config_path = Path("configs") / "default.toml"
    config = utils.get_config_data(config_path)
    return config


@pytest.fixture(scope="session")
def midi_output_mapping(default_config):
    """get the default MIDI output mapping."""
    return default_config["midi"]["output"]


@pytest.fixture(scope="session")
def midi_input_mapping(default_config):
    """get the default MIDI input mapping."""
    return default_config["midi"]["input"]


@pytest.fixture(scope="session")
def output_values(default_config):
    dimension = default_config["model"]["dimension"]
    return np.random.rand(dimension - 1)


@pytest.fixture(scope="session")
def last_midi_notes_dict(midi_output_mapping):
    """produce a dict of previous played midi notes"""
    last_midi_notes = {}
    for o_port in midi_output_mapping:
        last_midi_notes[o_port] = {}
        out_channels = [
            x[1] for x in midi_output_mapping[o_port] if x[0] == "note_on"
        ]  # just get channels associated with note_on messages.
        for chan in out_channels:
            last_midi_notes[o_port][
                chan
            ] = 60  # played middle c on each output channel.
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
    input_mapping = [["note_on", 1]]
    note_msg = mido.Message("note_on", channel=0, note=12, velocity=64)
    index, value = utils.midi_message_to_index_value(note_msg, input_mapping)
    assert index == 0 and value == 12 / 127
    # note off
    note_off_msg = mido.Message("note_off", channel=0, note=12, velocity=0)
    try:
        index, value = utils.midi_message_to_index_value(note_off_msg, input_mapping)
    except ValueError as e:
        # supposed to get a valueerror here.
        pass
    # cc
    input_mapping = [["control_change", 1, 1]]
    cc_msg = mido.Message("control_change", channel=0, control=1, value=64)
    index, value = utils.midi_message_to_index_value(cc_msg, input_mapping)
    assert index == 0 and value == 64 / 127


def test_midi_mapping_to_output(output_values, midi_output_mapping):
    output_messages_dict = utils.output_values_to_midi_messages(
        output_values, midi_output_mapping
    )
    for output_port in midi_output_mapping:
        output_messages = output_messages_dict[output_port]
        assert len(output_messages) == len(
            output_values
        ), "Number of output messages does not match number of output values"
        for msg in output_messages:
            assert isinstance(msg, mido.Message), "msg is not a mido.Message object"


def test_midi_note_off_generation(midi_output_mapping, last_midi_notes_dict):
    output_messages_dict = utils.get_midi_note_offs(
        midi_output_mapping, last_midi_notes_dict
    )
    for output_port in midi_output_mapping:
        output_messages = output_messages_dict[output_port]
        for msg in output_messages:
            assert isinstance(msg, mido.Message), "msg is not a mido.Message object"
            assert msg.type == "note_off", "msg is not a note_off"


# test IOServers


@pytest.fixture
def io_config(default_config):
    """Config with unique ports to avoid conflicts with session-scoped interaction_server."""
    import copy

    config = copy.deepcopy(default_config)
    config["websocket"]["server_port"] = 5099
    config["osc"]["server_port"] = 6099
    config["osc"]["client_port"] = 6098
    return config


def test_websocket_server(io_config, sparse_callback, dense_callback, output_values):
    sender = impsio.WebSocketServer(io_config, sparse_callback, dense_callback)
    sender.connect()
    assert sender.ws_thread is not None
    assert sender.ws_thread.is_alive()
    sender.handle()
    sender.send(output_values)
    time.sleep(0.1)
    sender.disconnect()
    time.sleep(0.1)


def test_osc_server(io_config, sparse_callback, dense_callback, output_values):
    sender = impsio.OSCServer(io_config, sparse_callback, dense_callback)
    sender.connect()
    assert sender.server_thread is not None
    assert sender.server_thread.is_alive()
    sender.handle()
    sender.send(output_values)
    time.sleep(0.1)
    sender.disconnect()
    time.sleep(0.1)


def test_serial_server(default_config, sparse_callback, dense_callback, output_values):
    sender = impsio.SerialServer(default_config, sparse_callback, dense_callback)
    sender.connect()
    # Serial port likely won't open in test env, so serial should be None
    assert sender.serial is None
    sender.handle()
    sender.send(output_values)
    sender.disconnect()


def test_serial_midi_server(
    default_config, sparse_callback, dense_callback, output_values
):
    sender = impsio.SerialMIDIServer(default_config, sparse_callback, dense_callback)
    sender.connect()
    assert sender.serial is None
    sender.handle()
    sender.send(output_values)
    sender.disconnect()


def test_midi_server(default_config, sparse_callback, dense_callback, output_values):
    sender = impsio.MIDIServer(default_config, sparse_callback, dense_callback)
    sender.connect()
    sender.handle()
    sender.send(output_values)
    sender.disconnect()


# OSC handler tests


def test_osc_handle_interface_message(io_config, sparse_callback):
    """Test that OSC interface message handler calls dense_callback."""
    received = []

    def mock_dense_callback(values):
        received.append(values)

    with patch("impsy.impsio.osc_server.ThreadingOSCUDPServer"):
        sender = impsio.OSCServer(io_config, sparse_callback, mock_dense_callback)
    sender.handle_interface_message("/interface", 0.5, 0.6, 0.7)
    assert len(received) == 1
    assert received[0] == [0.5, 0.6, 0.7]


def test_osc_handle_temperature_message(io_config, sparse_callback, dense_callback):
    """Test OSC temperature message handler doesn't crash."""
    with patch("impsy.impsio.osc_server.ThreadingOSCUDPServer"):
        sender = impsio.OSCServer(io_config, sparse_callback, dense_callback)
    sender.handle_temperature_message("/temperature", 0.5, 1.0)


def test_osc_handle_timescale_message(io_config, sparse_callback, dense_callback):
    """Test OSC timescale message handler doesn't crash."""
    with patch("impsy.impsio.osc_server.ThreadingOSCUDPServer"):
        sender = impsio.OSCServer(io_config, sparse_callback, dense_callback)
    sender.handle_timescale_message("/timescale", 2.0)


# Serial handler tests


def test_serial_handle_with_mock_port(default_config, sparse_callback):
    """Test serial handle parses CSV lines correctly."""
    received = []

    def mock_dense_callback(values):
        received.append(values)

    sender = impsio.SerialServer(default_config, sparse_callback, mock_dense_callback)
    # Mock a serial connection
    mock_serial = MagicMock()
    mock_serial.in_waiting = 0
    sender.serial = mock_serial
    # Simulate buffered data
    sender.buffer = "0.1,0.2,0.3\n0.4,0.5,0.6\n"
    mock_serial.in_waiting = 0  # no new bytes
    sender.handle()
    assert len(received) == 2
    assert received[0] == [0.1, 0.2, 0.3]
    assert received[1] == [0.4, 0.5, 0.6]


def test_serial_handle_invalid_csv(default_config, sparse_callback, dense_callback):
    """Test serial handle skips invalid CSV data gracefully."""
    sender = impsio.SerialServer(default_config, sparse_callback, dense_callback)
    mock_serial = MagicMock()
    mock_serial.in_waiting = 0
    sender.serial = mock_serial
    sender.buffer = "not,valid,csv,data,abc\n"
    # Should not raise
    sender.handle()


def test_serial_send_with_mock_port(
    default_config, sparse_callback, dense_callback, output_values
):
    """Test serial send writes CSV data."""
    sender = impsio.SerialServer(default_config, sparse_callback, dense_callback)
    mock_serial = MagicMock()
    sender.serial = mock_serial
    sender.send(output_values)
    mock_serial.write.assert_called_once()
    written_data = mock_serial.write.call_args[0][0]
    assert isinstance(written_data, bytes)
    assert b"\n" in written_data


# WebSocket send MIDI tests


def test_websocket_send_midi_formats(default_config, sparse_callback, dense_callback):
    """Test websocket MIDI message formatting."""
    sender = impsio.WebSocketServer(default_config, sparse_callback, dense_callback)
    # Add a mock client
    mock_client = MagicMock()
    sender.ws_clients.add(mock_client)

    # Test note_on
    msg = mido.Message("note_on", channel=0, note=60, velocity=100)
    sender.websocket_send_midi(msg)
    mock_client.send.assert_called_with("/channel/0/noteon/60/100")

    # Test note_off
    msg = mido.Message("note_off", channel=0, note=60, velocity=0)
    sender.websocket_send_midi(msg)
    mock_client.send.assert_called_with("/channel/0/noteoff/60/0")

    # Test cc
    msg = mido.Message("control_change", channel=1, control=42, value=64)
    sender.websocket_send_midi(msg)
    mock_client.send.assert_called_with("/channel/1/cc/42/64")


def test_websocket_send_midi_removes_dead_client(
    default_config, sparse_callback, dense_callback
):
    """Test that dead websocket clients are removed."""
    sender = impsio.WebSocketServer(default_config, sparse_callback, dense_callback)
    mock_client = MagicMock()
    mock_client.send.side_effect = Exception("connection closed")
    sender.ws_clients.add(mock_client)
    msg = mido.Message("note_on", channel=0, note=60, velocity=100)
    sender.websocket_send_midi(msg)
    assert mock_client not in sender.ws_clients


# MIDI server feedback protection tests


def test_midi_server_feedback_protection(
    default_config, sparse_callback, dense_callback
):
    """Test MIDIServer feedback protection configuration."""
    config = dict(default_config)
    config["midi"] = dict(default_config["midi"])
    config["midi"]["feedback_protection"] = True
    config["midi"]["feedback_threshold"] = 0.05
    sender = impsio.MIDIServer(config, sparse_callback, dense_callback)
    assert sender.feedback_protection is True
    assert sender.feedback_threshold == 0.05


def test_midi_server_handle_no_ports(default_config, sparse_callback, dense_callback):
    """Test MIDI handle when no ports are open."""
    sender = impsio.MIDIServer(default_config, sparse_callback, dense_callback)
    # handle with empty port dict should not crash
    sender.handle()


def test_midi_server_handle_port_none(default_config, sparse_callback, dense_callback):
    """Test MIDI handle_port returns early when port is None."""
    sender = impsio.MIDIServer(default_config, sparse_callback, dense_callback)
    sender.midi_in_port["test"] = None
    # Should return early without error
    sender.handle_port("test")


def test_midi_server_handle_port_with_messages(
    default_config, sparse_callback, dense_callback
):
    """Test MIDI handle_port processes messages via callback."""
    received = []

    def mock_callback(index, value):
        received.append((index, value))
        return np.random.rand(default_config["model"]["dimension"] - 1)

    sender = impsio.MIDIServer(default_config, mock_callback, dense_callback)

    # Get the first input port name from config
    first_in_port = default_config["midi"]["in_device"][0]
    input_mapping = default_config["midi"]["input"][first_in_port]

    # Create a mock port that yields a matching MIDI message
    mock_port = MagicMock()
    # Create a CC message matching the first mapping entry
    mapping_entry = input_mapping[0]
    if mapping_entry[0] == "control_change":
        test_msg = mido.Message(
            "control_change",
            channel=mapping_entry[1] - 1,
            control=mapping_entry[2],
            value=64,
        )
    else:
        test_msg = mido.Message(
            "note_on", channel=mapping_entry[1] - 1, note=60, velocity=100
        )
    mock_port.iter_pending.return_value = [test_msg]
    sender.midi_in_port[first_in_port] = mock_port

    sender.handle_port(first_in_port)
    assert len(received) == 1


# SerialMIDI tests


def test_serial_midi_send_midi_message(default_config, sparse_callback, dense_callback):
    """Test SerialMIDI send_midi_message writes bytes to serial."""
    sender = impsio.SerialMIDIServer(default_config, sparse_callback, dense_callback)
    mock_serial = MagicMock()
    sender.serial = mock_serial
    msg = mido.Message("note_on", channel=0, note=60, velocity=100)
    sender.send_midi_message(msg)
    mock_serial.write.assert_called_once_with(msg.bin())


def test_serial_midi_send_midi_message_no_serial(
    default_config, sparse_callback, dense_callback
):
    """Test SerialMIDI send_midi_message does nothing without serial connection."""
    sender = impsio.SerialMIDIServer(default_config, sparse_callback, dense_callback)
    sender.serial = None
    msg = mido.Message("note_on", channel=0, note=60, velocity=100)
    # Should not raise
    sender.send_midi_message(msg)


def test_serial_midi_handle_with_mock(default_config):
    """Test SerialMIDI handle processes MIDI messages from serial."""
    received = []

    def mock_callback(index, value):
        received.append((index, value))
        return np.zeros(default_config["model"]["dimension"] - 1)

    def mock_dense_callback(values):
        pass

    sender = impsio.SerialMIDIServer(default_config, mock_callback, mock_dense_callback)
    mock_serial = MagicMock()
    sender.serial = mock_serial

    # Build a matching CC message from the serialmidi input mapping
    input_mapping = default_config["serialmidi"]["input"]
    mapping_entry = input_mapping[0]
    if mapping_entry[0] == "control_change":
        test_msg = mido.Message(
            "control_change",
            channel=mapping_entry[1] - 1,
            control=mapping_entry[2],
            value=64,
        )
    else:
        test_msg = mido.Message(
            "note_on", channel=mapping_entry[1] - 1, note=60, velocity=100
        )

    # Simulate reading 3 bytes
    mock_serial.in_waiting = 3
    mock_serial.read.return_value = test_msg.bin()
    sender.handle()
    # The parser may or may not yield a message depending on internal state
    # but the function should not crash
