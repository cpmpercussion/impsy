import numpy as np
import pandas as pd
import tomllib
import click
import mido
from typing import List


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
    r_data = np.random.normal(size=NSAMPLE)
    # x_data = np.sin(t_data) * 1.0 + (r_data * 0.05)
    df = pd.DataFrame({"t": t_data})
    for i in range(dimension - 1):
        df[f"x{i}"] = df["t"].apply(fuzzy_sine_function, scale=i)
    df.t = df.t.diff()
    df.t = df.t.fillna(1e-4)
    print(df.describe())
    return np.array(df)


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
        click.secho(f"Error: Configuration file '{config_path}' is not valid TOML format.", fg="red")
        raise click.Abort()
    return config_data
    

# MIDI mapping and message utilities


def output_values_to_midi_messages(output_values: List[float], midi_mapping: dict) -> List[mido.Message]:
    """Transforms a list of output values to a list of MIDI messages using a mapping."""
    output_messages = []
    output_midi = list(map(int, (np.ceil(output_values * 127)))) # transform output values to MIDI 0-127.

    for i in range(len(output_values)):
        if midi_mapping[i][0] == "note_on":
            # note decremented channel (0-15)
            # note velocity is maximum at 127
            midi_msg = mido.Message("note_on", channel=midi_mapping[i][1] - 1, note=output_midi[i], velocity=127)
            output_messages.append(midi_msg)
        if midi_mapping[i][0] == "control_change":
            # note decremented channel (0-15)
            # note control number starts at 0
            midi_msg = mido.Message("control_change", channel=midi_mapping[i][1] - 1, control=midi_mapping[i][2], value=output_midi[i])
            output_messages.append(midi_msg)
    # return the MIDI messages
    return output_messages


def get_midi_note_offs(midi_mapping: dict, last_midi_notes: dict) -> List[mido.Message]:
    """Get a list of note_off messages for any MIDI channels that have been used for notes."""
    output_messages = []
    out_channels = [x[1] for x in midi_mapping if x[0] == "note_on"] # just get channels associated with note_on messages.
    for i in out_channels:
        channel = i - 1 # decrement to get channel value 0-15
        if channel in last_midi_notes:
            midi_msg = mido.Message("note_off", channel=i - 1,  note=last_midi_notes[channel], velocity=0)
            output_messages.append(midi_msg)
    return output_messages


def midi_message_to_index_value(msg: mido.Message, input_mapping: dict) -> (int, float):
    """Takes a MIDO message and an input mapping and returns a tuple of index and value for sending to the IMPSY callback."""
    if msg.type == "note_on":
        index = input_mapping.index(
            ["note_on", msg.channel + 1]
        )
        value = msg.note / 127.0
    elif msg.type == "control_change":
        index = input_mapping.index(
            ["control_change", msg.channel + 1, msg.control]
        )
        value = msg.value / 127.0
    else:
        raise ValueError(f"Only note_on and control_change messages can be processed, this was a {msg.type} message.")
    return (index, value)


def match_midi_port_to_list(port, port_list):
    """Return the closest actual MIDI port name given a partial match and a list."""
    if port in port_list:
        return port
    contains_list = [x for x in port_list if port in x]
    if not contains_list:
        return False
    else:
        return contains_list[0]
    

# Printing functions

def print_io(label, values, colour):
    """Neatly prints an array of values with a label."""
    vals = np.array([round(val, 3) for val in values])
    click.secho(f"{label}: {vals}", fg=colour)