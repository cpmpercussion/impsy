from impsy import impsio, utils
import pytest
import numpy as np
import time
from pathlib import Path


@pytest.fixture(scope="session")
def default_config():
    """get the default config file."""
    config_path = Path("configs") / "default.toml"
    config = utils.get_config_data(config_path)
    return(config)


@pytest.fixture(scope="session")
def output_values(default_config):
    dimension = default_config["model"]["dimension"]
    return np.random.rand(dimension - 1)


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
    sender = impsio.SerialMIDIServer(
        default_config, sparse_callback, dense_callback
    )
    sender.connect()
    # sender.handle()
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
