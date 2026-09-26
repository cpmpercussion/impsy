import numpy as np
import tomllib
import click
import mido
from typing import List, Dict

# MDRNN config


SIZE_TO_PARAMETERS = {
    "xxs": {
        "units": 16,
        "mixes": 5,
        "layers": 2,
    },
    "xs": {
        "units": 32,
        "mixes": 5,
        "layers": 2,
    },
    "s": {"units": 64, "mixes": 5, "layers": 2},
    "m": {"units": 128, "mixes": 5, "layers": 2},
    "l": {"units": 256, "mixes": 5, "layers": 2},
    "xl": {"units": 512, "mixes": 5, "layers": 3},
    "default": {"units": 128, "mixes": 5, "layers": 2},
}


def mdrnn_config(size: str):
    """Get a config dictionary from a size string as used in the IMPS command line interface."""
    return SIZE_TO_PARAMETERS[size]


# Fake data generator for tests.


def fuzzy_sine_function(t, scale=1.0, fuzz_factor=0.05):
    """A fuzzy sine function with variable fuzz factor"""
    return np.sin(t) * scale + (np.random.normal() * fuzz_factor)


def generate_data(samples: int = 50000, dimension: int = 2):
    """Generating some Slightly fuzzy sine wave data."""
    assert dimension > 1, "dimension must be greater than 1"
    NSAMPLE = samples
    print("Generating", str(NSAMPLE), "toy data samples.")
    t_data = np.float32(np.array(range(NSAMPLE)) / 10.0)
    t_interval = t_data[1] - t_data[0]
    t_r_data = np.random.normal(
        0, t_interval / 20.0, size=NSAMPLE
    )  ## fuzz up the time sampling
    t_data = t_data + t_r_data

    # Build columns: t, x0, x1, ..., x(n-2)
    columns = np.zeros((NSAMPLE, dimension), dtype=np.float32)
    for i in range(dimension - 1):
        columns[:, i + 1] = np.array(
            [fuzzy_sine_function(t, scale=i) for t in t_data], dtype=np.float32
        )

    # Compute time diffs
    dt = np.diff(t_data, prepend=t_data[0])
    dt[0] = 1e-4
    columns[:, 0] = dt

    return columns


def get_config_data(config_path: str):
    """Loads a TOML config from a string path."""
    click.secho(f"Opening configuration from {config_path}", fg="yellow")
    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
    except FileNotFoundError:
        click.secho(f"Error: Could not find config file '{config_path}'.", fg="red")
        raise click.Abort()
    except tomllib.TOMLDecodeError:
        click.secho(
            f"Error: Configuration file '{config_path}' is not valid TOML format.",
            fg="red",
        )
        raise click.Abort()
    return config_data


# MIDI mapping and message utilities


def value_to_midi(value: float) -> int:
    """Quantise a value in [0, 1] to a MIDI data byte 0-127, rounding to nearest (half up)."""
    return int(np.clip(np.floor(float(value) * 127 + 0.5), 0, 127))


def process_midi_min_max(value: int, min_value: int, max_value: int) -> int:
    """Process a MIDI control change value to fit within a min and max range."""
    range = max_value - min_value
    return int(np.floor(range * value / 127 + 0.5) + min_value)


class MidiOutputState:
    """Turns output vectors into MIDI messages for one output mapping.

    Notes are tracked per dimension, so several note_on dimensions on the same
    channel can sound together. Before a dimension plays a new note, its
    previous note is turned off, unless another dimension on that channel is
    still holding the same note.
    """

    def __init__(self, mapping: list):
        self.mapping = mapping
        self.sounding = {}  # dimension index -> (channel, note), 0-based channel
        self.last_note_on = {}  # channel -> most recent note sent on it

    def _held_elsewhere(self, index: int, channel_note: tuple) -> bool:
        return any(
            held == channel_note for i, held in self.sounding.items() if i != index
        )

    def messages(self, output_values) -> List[mido.Message]:
        messages = []
        for i, entry in enumerate(self.mapping):
            if i >= len(output_values):
                break
            channel = entry[1] - 1
            midi_value = value_to_midi(output_values[i])
            if entry[0] == "note_on":
                previous = self.sounding.pop(i, None)
                if previous is not None and not self._held_elsewhere(i, previous):
                    messages.append(
                        mido.Message(
                            "note_off",
                            channel=previous[0],
                            note=previous[1],
                            velocity=0,
                        )
                    )
                messages.append(
                    mido.Message(
                        "note_on", channel=channel, note=midi_value, velocity=127
                    )
                )
                self.sounding[i] = (channel, midi_value)
                self.last_note_on[channel] = midi_value
            elif entry[0] == "control_change":
                if len(entry) == 5:
                    midi_value = process_midi_min_max(midi_value, entry[3], entry[4])
                messages.append(
                    mido.Message(
                        "control_change",
                        channel=channel,
                        control=entry[2],
                        value=midi_value,
                    )
                )
        return messages

    def all_notes_off(self) -> List[mido.Message]:
        """Note-offs for every sounding note, e.g. on disconnect."""
        messages = [
            mido.Message("note_off", channel=channel, note=note, velocity=0)
            for channel, note in dict.fromkeys(self.sounding.values())
        ]
        self.sounding = {}
        return messages


def midi_message_to_indices_value(
    msg: mido.Message, input_mapping: list
) -> (List[int], float):
    """Takes a MIDO message and an input mapping and returns the indices it maps to and its value.

    A message can map to several dimensions; all of them get the same value.
    Note-ons with velocity 0 are note-offs and, like other messages that
    don't change the input, raise ValueError.
    """
    if msg.type == "note_on":
        if msg.velocity == 0:
            raise ValueError("Note-ons with velocity 0 are note-offs.")
        key = ["note_on", msg.channel + 1]
        value = msg.note / 127.0
    elif msg.type == "control_change":
        key = ["control_change", msg.channel + 1, msg.control]
        value = msg.value / 127.0
    else:
        raise ValueError(
            f"Only note_on and control_change messages can be processed, this was a {msg.type} message."
        )
    indices = [i for i, entry in enumerate(input_mapping) if list(entry) == key]
    if not indices:
        raise ValueError(f"No input mapping for {msg}.")
    return (indices, value)


def match_midi_port_to_list(port, port_list, verbose=True):
    """Return the closest actual MIDI port name given a partial match and a list."""
    if verbose:
        click.secho(
            f"Matching MIDI port '{port}' to available ports {port_list}", fg="blue"
        )
    if port in port_list:
        return port
    contains_list = [x for x in port_list if port in x]
    if verbose:
        click.secho(f"Found matching ports: {contains_list}", fg="blue")
    if not contains_list:
        return False
    else:
        return contains_list[0]


# Printing functions


def print_io(label, values, colour):
    """Neatly prints an array of values with a label."""
    vals = np.array([round(val, 3) for val in values])
    click.secho(f"{label}: {vals}", fg=colour)
