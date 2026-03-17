"""Tests for impsy.utils module."""

from impsy import utils
import pytest
import click
import mido
import numpy as np
from pathlib import Path


def test_get_config_missing_file():
    """Test that get_config_data raises Abort for missing file."""
    with pytest.raises(click.Abort):
        utils.get_config_data("nonexistent_config.toml")


def test_get_config_invalid_toml(tmp_path):
    """Test that get_config_data raises Abort for invalid TOML."""
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("this is [not valid toml = = =")
    with pytest.raises(click.Abort):
        utils.get_config_data(str(bad_toml))


def test_get_config_valid(tmp_path):
    """Test that get_config_data loads a valid TOML file."""
    good_toml = tmp_path / "good.toml"
    good_toml.write_text('[model]\ndimension = 4\nsize = "xs"\n')
    config = utils.get_config_data(str(good_toml))
    assert config["model"]["dimension"] == 4


def test_mdrnn_config_sizes():
    """Test that all size strings return valid configs."""
    for size in ["xxs", "xs", "s", "m", "l", "xl"]:
        config = utils.mdrnn_config(size)
        assert "units" in config
        assert "mixes" in config
        assert "layers" in config
        assert config["units"] > 0
        assert config["mixes"] > 0
        assert config["layers"] > 0


def test_mdrnn_config_invalid_size():
    """Test that invalid size raises KeyError."""
    with pytest.raises(KeyError):
        utils.mdrnn_config("invalid_size")


def test_process_midi_min_max():
    """Test MIDI min/max processing with boundary values."""
    # No scaling (full range)
    assert utils.process_midi_min_max(0, 0, 127) == 0
    assert utils.process_midi_min_max(127, 0, 127) == 127

    # Restricted range
    result = utils.process_midi_min_max(64, 10, 100)
    assert 10 <= result <= 100

    # Min equals max (degenerate case)
    result = utils.process_midi_min_max(64, 50, 50)
    assert result == 50


def test_midi_message_to_index_value_unsupported_type():
    """Test that unsupported MIDI message types raise ValueError."""
    msg = mido.Message("pitchwheel", channel=0, pitch=0)
    with pytest.raises(ValueError, match="Only note_on and control_change"):
        utils.midi_message_to_index_value(msg, [["note_on", 1]])


def test_match_midi_port_exact():
    """Test exact MIDI port matching."""
    port_list = ["IAC Driver Bus 1", "USB MIDI Device"]
    result = utils.match_midi_port_to_list("IAC Driver Bus 1", port_list, verbose=False)
    assert result == "IAC Driver Bus 1"


def test_match_midi_port_partial():
    """Test partial MIDI port matching."""
    port_list = ["IAC Driver Bus 1", "USB MIDI Device Port 0"]
    result = utils.match_midi_port_to_list("USB MIDI", port_list, verbose=False)
    assert result == "USB MIDI Device Port 0"


def test_match_midi_port_not_found():
    """Test MIDI port matching when no match exists."""
    port_list = ["IAC Driver Bus 1"]
    result = utils.match_midi_port_to_list("NonExistent", port_list, verbose=False)
    assert result is False


def test_generate_data():
    """Test synthetic data generation."""
    data = utils.generate_data(samples=100, dimension=4)
    assert data.shape == (100, 4)


def test_generate_data_minimum_dimension():
    """Test that dimension must be > 1."""
    with pytest.raises(AssertionError):
        utils.generate_data(samples=10, dimension=1)
