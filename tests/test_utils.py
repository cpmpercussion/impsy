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


def test_value_to_midi_with_range():
    """Test scaling values into a MIDI min/max range, rounding once."""
    assert utils.value_to_midi(0.0, 0, 127) == 0
    assert utils.value_to_midi(1.0, 0, 127) == 127
    assert utils.value_to_midi(0.0, 10, 100) == 10
    assert utils.value_to_midi(1.0, 10, 100) == 100
    assert utils.value_to_midi(0.008, 0, 63) == 1  # 0.504
    # Min equals max (degenerate case)
    assert utils.value_to_midi(0.5, 50, 50) == 50


def test_midi_to_value_inverts_range():
    assert utils.midi_to_value(64, 0, 127) == 64 / 127
    assert utils.midi_to_value(21, 0, 63) == 21 / 63
    assert utils.midi_to_value(100, 0, 63) == 1.0  # clamped
    assert utils.midi_to_value(0, 64, 127) == 0.0  # clamped
    assert utils.midi_to_value(5, 50, 50) == 0.0
    for midi in range(64, 128):
        assert utils.value_to_midi(utils.midi_to_value(midi, 64, 127), 64, 127) == midi


def test_midi_message_to_updates_unsupported_type():
    """Test that unsupported MIDI message types raise ValueError."""
    msg = mido.Message("aftertouch", channel=0, value=64)
    with pytest.raises(ValueError, match="Only note_on, control_change and pitchwheel"):
        utils.midi_message_to_updates(msg, [["note_on", 1]])


def test_note_on_sets_pitch_and_velocity_dimensions():
    mapping = [["note_on", 1], ["note_velocity", 1], ["note_velocity", 2]]
    msg = mido.Message("note_on", channel=0, note=60, velocity=100)
    assert utils.midi_message_to_updates(msg, mapping) == [
        (0, 60 / 127),
        (1, 100 / 127),
    ]
    # a fixed-velocity note mapping still matches on input
    assert utils.midi_message_to_updates(msg, [["note_on", 1, 90]]) == [(0, 60 / 127)]


def test_pitch_bend_round_trip():
    assert utils.pitch_bend_to_value(-8192) == 0.0
    assert utils.pitch_bend_to_value(8191) == 1.0
    assert utils.value_to_pitch_bend(0.5) == 0
    assert utils.value_to_pitch_bend(-1.0) == -8192
    assert utils.value_to_pitch_bend(2.0) == 8191
    for pitch in range(-8192, 8192, 7):
        assert utils.value_to_pitch_bend(utils.pitch_bend_to_value(pitch)) == pitch
    msg = mido.Message("pitchwheel", channel=2, pitch=8191)
    assert utils.midi_message_to_updates(msg, [["note_on", 3], ["pitch_bend", 3]]) == [
        (1, 1.0)
    ]


def test_output_velocity_sources():
    """Velocity comes from a note_velocity dimension, then a fixed velocity, then 127."""
    state = utils.MidiOutputState(
        [["note_on", 1], ["note_velocity", 1], ["note_on", 2, 90], ["note_on", 3]]
    )
    velocities = [
        m.velocity for m in state.messages([0.5, 0.5, 0.5, 0.5]) if m.type == "note_on"
    ]
    assert velocities == [64, 90, 127]
    # velocity 0 would be a note-off, so the lowest velocity sent is 1
    velocities = [
        m.velocity for m in state.messages([0.5, 0.0, 0.5, 0.5]) if m.type == "note_on"
    ]
    assert velocities[0] == 1


def test_velocity_dimension_overrides_fixed_velocity():
    state = utils.MidiOutputState([["note_on", 1, 90], ["note_velocity", 1]])
    (note_on,) = state.messages([0.5, 1.0])
    assert note_on.velocity == 127


def test_pitch_bend_output():
    state = utils.MidiOutputState([["pitch_bend", 2]])
    (msg,) = state.messages([1.0])
    assert msg.type == "pitchwheel" and msg.channel == 1 and msg.pitch == 8191


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


def test_unchanged_cc_and_pitch_bend_not_resent():
    state = utils.MidiOutputState(
        [["note_on", 1], ["control_change", 1, 7], ["pitch_bend", 1]]
    )
    first = state.messages([0.5, 0.5, 0.5])
    assert [m.type for m in first] == ["note_on", "control_change", "pitchwheel"]
    # same values: only the note (with its note-off) is sent again
    again = state.messages([0.5, 0.5, 0.5])
    assert [m.type for m in again] == ["note_off", "note_on"]
    # a change sends just that controller
    changed = state.messages([0.5, 0.9, 0.5])
    assert [m.type for m in changed] == ["note_off", "note_on", "control_change"]
    # all-notes-off forgets the values, so everything is sent again
    state.all_notes_off()
    assert len(state.messages([0.5, 0.9, 0.5])) == 3
