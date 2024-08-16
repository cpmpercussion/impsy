from impsy import interaction
from impsy import utils
import pytest
from pathlib import Path
import time
import numpy as np
import logging

@pytest.fixture(scope="session")
def default_config():
    """get the default config file."""
    config_path = Path("configs") / "default.toml"
    config = utils.get_config_data(config_path)
    return(config)


@pytest.fixture(scope="session")
def dimension(default_config):
    return default_config["model"]["dimension"]

@pytest.fixture(scope="session")
def log_location(tmp_path_factory):
    location = tmp_path_factory.mktemp("logs")
    return location

@pytest.fixture(scope="session")
def logger(dimension, log_location):
    logger = interaction.setup_logging(dimension, location=log_location)
    return logger


def test_logging(logger, dimension):
    """Just sets up logging"""
    assert isinstance(logger, logging.Logger)
    values = np.random.rand(dimension - 1)
    interaction.log_interaction("tests", values, logger)
    interaction.close_log(logger)


@pytest.fixture(scope="session")
def neural_network(default_config):
    net = interaction.build_network(default_config)
    return net


def test_build_network(neural_network):
    pass


@pytest.fixture(scope="session")
def interaction_server(default_config, log_location):
    interaction_server = interaction.InteractionServer(default_config, log_location=log_location)
    return interaction_server

# def test_broken_interaction_server():
#     interaction_server = interaction.InteractionServer({})

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
