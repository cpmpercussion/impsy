from impsy import interaction
from impsy import utils
import pytest
from pathlib import Path
import time
import numpy as np

@pytest.fixture(scope="session")
def default_config():
    """get the default config file."""
    config_path = Path("configs") / "default.toml"
    config = utils.get_config_data(config_path)
    return(config)


@pytest.fixture(scope="session")
def dimension(default_config):
    return default_config["model"]["dimension"]


def test_logging(dimension):
    """Just sets up logging"""
    interaction.setup_logging(dimension)


@pytest.fixture(scope="session")
def neural_network(default_config):
    net = interaction.build_network(default_config)
    return net


def test_build_network(neural_network):
    pass


@pytest.fixture(scope="session")
def interaction_server(default_config):
    interaction_server = interaction.InteractionServer(default_config)
    return interaction_server


def test_monitor_user_action(interaction_server):
    """Just tests creation of an interaction server object"""
    interaction_server.monitor_user_action()


def test_make_prediction(interaction_server, neural_network):
    interaction_server.make_prediction(neural_network)


def test_input_list(interaction_server):
    interaction_server.construct_input_list(0,0.0)


def test_dense_callback(interaction_server, dimension):
    values = np.random.rand(dimension - 1)
    interaction_server.dense_callback(values)


def test_send_values(interaction_server, dimension):
    values = np.random.rand(dimension - 1)
    interaction_server.send_back_values(values)
