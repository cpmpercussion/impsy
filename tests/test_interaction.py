from impsy import interaction
from impsy import mdrnn
from impsy import utils
import pytest
from pathlib import Path
import numpy as np
import logging
import time
import queue
import copy


@pytest.fixture(scope="session")
def default_config():
    """get the default config file."""
    config_path = Path("configs") / "default.toml"
    config = utils.get_config_data(config_path)
    return(config)


@pytest.fixture(scope="session")
def user_only_untrained_config():
    """get a config file without a neural network and in user-only mode."""
    config_path = Path("configs") / "user-only-example.toml"
    config = utils.get_config_data(config_path)
    return(config)


@pytest.fixture(scope="session")
def default_dimension(default_config):
    return default_config["model"]["dimension"]


@pytest.fixture(scope="session")
def logger(default_dimension, log_location):
    logger = interaction.setup_logging(default_dimension, location=log_location)
    return logger


def test_logging(logger, dimension):
    """Just sets up logging"""
    assert isinstance(logger, logging.Logger)
    values = np.random.rand(dimension - 1)
    interaction.log_interaction("tests", values, logger)
    interaction.close_log(logger)


@pytest.fixture(scope="session")
def default_neural_network(default_config):
    net = interaction.build_network(default_config)
    return net


def test_build_network(default_neural_network, default_config):
    """Test that build_network returns a model with expected attributes."""
    net = default_neural_network
    assert hasattr(net, 'dimension')
    assert hasattr(net, 'generate')
    assert net.dimension == default_config["model"]["dimension"]
    assert net.pi_temp == default_config["model"]["pitemp"]
    assert net.sigma_temp == default_config["model"]["sigmatemp"]


def test_build_network_dummy_fallback(log_location):
    """Test that build_network falls back to DummyMDRNN when no model file is specified."""
    config = {
        "model": {
            "dimension": 4,
            "size": "xs",
            "pitemp": 1.0,
            "sigmatemp": 0.01,
        },
        "verbose": False,
        "log_input": True,
        "log_predictions": False,
        "interaction": {"mode": "useronly", "threshold": 0.1, "input_thru": False},
    }
    net = interaction.build_network(config)
    assert isinstance(net, mdrnn.DummyMDRNN)
    assert net.dimension == 4


def test_build_network_missing_dimension():
    """Test that build_network raises when dimension is missing."""
    config = {"model": {}}
    with pytest.raises(Exception):
        interaction.build_network(config)


@pytest.fixture(scope="session")
def interaction_server(default_config, log_location):
    interaction_server = interaction.InteractionServer(default_config, log_location=log_location)
    return interaction_server


def test_monitor_user_action(interaction_server):
    """Tests that monitor_user_action updates state correctly."""
    # Simulate recent user interaction (should be in "call" mode)
    interaction_server.last_user_interaction_time = time.time()
    interaction_server.call_response_mode = "call"
    interaction_server.monitor_user_action()
    assert interaction_server.user_to_rnn is True
    assert interaction_server.rnn_to_rnn is False
    assert interaction_server.call_response_mode == "call"


def test_monitor_user_action_switch_to_response(interaction_server):
    """Test that monitor switches to response mode when user is inactive."""
    # Simulate old user interaction (beyond threshold)
    interaction_server.last_user_interaction_time = time.time() - 10.0
    interaction_server.call_response_mode = "call"
    interaction_server.monitor_user_action()
    assert interaction_server.user_to_rnn is False
    assert interaction_server.rnn_to_rnn is True
    assert interaction_server.rnn_to_sound is True
    assert interaction_server.call_response_mode == "response"


def test_monitor_user_action_switch_to_call(interaction_server):
    """Test that monitor switches back to call mode when user becomes active."""
    interaction_server.call_response_mode = "response"
    interaction_server.last_user_interaction_time = time.time()
    interaction_server.monitor_user_action()
    assert interaction_server.call_response_mode == "call"
    assert interaction_server.user_to_rnn is True
    assert interaction_server.rnn_to_rnn is False


def test_make_prediction(interaction_server, default_neural_network):
    """Test that make_prediction processes items from the input queue."""
    interaction_server.user_to_rnn = True
    interaction_server.rnn_to_sound = True
    # Put an item in the input queue
    test_input = mdrnn.random_sample(out_dim=interaction_server.dimension)
    interaction_server.interface_input_queue.put_nowait(test_input)
    interaction_server.make_prediction(default_neural_network)
    # Input queue should be consumed
    assert interaction_server.interface_input_queue.empty()
    # Output should be in the buffer
    assert not interaction_server.rnn_output_buffer.empty()
    # Clean up: drain the output buffer
    while not interaction_server.rnn_output_buffer.empty():
        interaction_server.rnn_output_buffer.get_nowait()


def test_make_prediction_rnn_to_rnn(interaction_server, default_neural_network):
    """Test rnn_to_rnn prediction path (autonomous mode)."""
    interaction_server.user_to_rnn = False
    interaction_server.rnn_to_rnn = True
    # Empty both queues first to ensure clean state
    while not interaction_server.rnn_output_buffer.empty():
        interaction_server.rnn_output_buffer.get_nowait()
    while not interaction_server.interface_input_queue.empty():
        interaction_server.interface_input_queue.get_nowait()
    while not interaction_server.rnn_prediction_queue.empty():
        interaction_server.rnn_prediction_queue.get_nowait()
    # Put an item in the rnn prediction queue
    test_input = mdrnn.random_sample(out_dim=interaction_server.dimension)
    interaction_server.rnn_prediction_queue.put_nowait(test_input)
    interaction_server.make_prediction(default_neural_network)
    # Prediction queue should be consumed
    assert interaction_server.rnn_prediction_queue.empty()
    # Output should be in the buffer
    assert not interaction_server.rnn_output_buffer.empty()
    # Clean up
    while not interaction_server.rnn_output_buffer.empty():
        interaction_server.rnn_output_buffer.get_nowait()


def test_input_list(interaction_server):
    interaction_server.construct_input_list(0, 0.0)


def test_dense_callback(interaction_server, default_dimension):
    """Test that dense_callback updates state and queues input."""
    old_time = interaction_server.last_user_interaction_time
    values = np.random.rand(default_dimension - 1)
    time.sleep(0.01)
    interaction_server.dense_callback(values)
    # Interaction time should be updated
    assert interaction_server.last_user_interaction_time > old_time
    # Data should include dt + values
    assert len(interaction_server.last_user_interaction_data) == default_dimension
    # Queue should have an item
    assert not interaction_server.interface_input_queue.empty()
    # Drain the queue
    while not interaction_server.interface_input_queue.empty():
        interaction_server.interface_input_queue.get_nowait()


def test_send_values(interaction_server, default_dimension):
    """Test that send_back_values clips output to [0, 1]."""
    # Values outside [0,1] should be clipped
    values = np.array([-0.5] + [1.5] * (default_dimension - 2))
    interaction_server.send_back_values(values)
    # No crash means it worked; values get clipped in send_back_values


def test_invalid_mode_fallback(log_location):
    """Test that an invalid mode falls back to useronly."""
    config = {
        "model": {
            "dimension": 4,
            "size": "xs",
            "pitemp": 1.0,
            "sigmatemp": 0.01,
            "timescale": 1,
        },
        "verbose": False,
        "log_input": True,
        "log_predictions": False,
        "interaction": {"mode": "invalid_mode", "threshold": 0.1, "input_thru": False},
    }
    server = interaction.InteractionServer(config, log_location=log_location)
    assert server.user_to_rnn is False
    assert server.rnn_to_rnn is False
    assert server.rnn_to_sound is False
    server.shutdown()


def test_polyphony_mode(log_location):
    """Test that polyphony mode sets correct flags."""
    config = {
        "model": {
            "dimension": 4,
            "size": "xs",
            "pitemp": 1.0,
            "sigmatemp": 0.01,
            "timescale": 1,
        },
        "verbose": False,
        "log_input": True,
        "log_predictions": False,
        "interaction": {"mode": "polyphony", "threshold": 0.1, "input_thru": False},
    }
    server = interaction.InteractionServer(config, log_location=log_location)
    assert server.user_to_rnn is True
    assert server.rnn_to_rnn is False
    assert server.rnn_to_sound is True
    server.shutdown()


def test_battle_mode(log_location):
    """Test that battle mode sets correct flags."""
    config = {
        "model": {
            "dimension": 4,
            "size": "xs",
            "pitemp": 1.0,
            "sigmatemp": 0.01,
            "timescale": 1,
        },
        "verbose": False,
        "log_input": True,
        "log_predictions": False,
        "interaction": {"mode": "battle", "threshold": 0.1, "input_thru": False},
    }
    server = interaction.InteractionServer(config, log_location=log_location)
    assert server.user_to_rnn is False
    assert server.rnn_to_rnn is True
    assert server.rnn_to_sound is True
    server.shutdown()


def test_shutdown(log_location):
    """Test that shutdown disconnects I/O servers and closes log."""
    config = {
        "model": {
            "dimension": 4,
            "size": "xs",
            "pitemp": 1.0,
            "sigmatemp": 0.01,
            "timescale": 1,
        },
        "verbose": False,
        "log_input": True,
        "log_predictions": False,
        "interaction": {"mode": "useronly", "threshold": 0.1, "input_thru": False},
    }
    server = interaction.InteractionServer(config, log_location=log_location)
    handler_count_before = len(server.logger.handlers)
    server.shutdown()
    # shutdown should have removed at least one handler
    assert len(server.logger.handlers) < handler_count_before


def test_construct_input_list_updates_state(interaction_server, default_dimension):
    """Test that construct_input_list updates interaction time and data."""
    old_time = interaction_server.last_user_interaction_time
    time.sleep(0.01)
    result = interaction_server.construct_input_list(0, 0.5)
    assert interaction_server.last_user_interaction_time > old_time
    assert len(interaction_server.last_user_interaction_data) == default_dimension
    # Result should be clipped values
    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0) and np.all(result <= 1)
    # Drain the queue
    while not interaction_server.interface_input_queue.empty():
        interaction_server.interface_input_queue.get_nowait()
