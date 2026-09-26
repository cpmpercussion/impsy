import numpy as np
import tomllib
import click
import mido
from typing import List, Dict, Tuple

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


def value_to_midi(value: float, min_value: int = 0, max_value: int = 127) -> int:
    """Scale a value in [0, 1] to a MIDI data byte in [min_value, max_value], rounding to nearest (half up)."""
    value = min(max(float(value), 0.0), 1.0)
    scaled = min_value + value * (max_value - min_value)
    return int(np.clip(np.floor(scaled + 0.5), 0, 127))


def midi_to_value(midi_value: int, min_value: int = 0, max_value: int = 127) -> float:
    """The inverse of value_to_midi: a MIDI data byte in [min_value, max_value] to a value in [0, 1].

    Bytes outside the range are clamped to it; an empty range gives 0.
    """
    if min_value == max_value:
        return 0.0
    low, high = sorted((min_value, max_value))
    clamped = min(max(midi_value, low), high)
    return (clamped - min_value) / (max_value - min_value)


PITCH_BEND_RANGE = 16383  # 14-bit pitch bend, 0-16383 centred at 8192
PITCH_BEND_CENTRE = 8192


def pitch_bend_to_value(pitch: int) -> float:
    """A mido pitchwheel value (-8192 to 8191) to a value in [0, 1]."""
    return (pitch + PITCH_BEND_CENTRE) / PITCH_BEND_RANGE


def value_to_pitch_bend(value: float) -> int:
    """A value in [0, 1] to a mido pitchwheel value (-8192 to 8191), rounding to nearest (half up)."""
    value = min(max(float(value), 0.0), 1.0)
    return int(np.floor(value * PITCH_BEND_RANGE + 0.5)) - PITCH_BEND_CENTRE


class MidiOutputState:
    """Turns output vectors into MIDI messages for one output mapping.

    Notes are tracked per dimension, so several note_on dimensions on the same
    channel can sound together. Before a dimension plays a new note, its
    previous note is turned off, unless another dimension on that channel is
    still holding the same note.

    A note's velocity comes from the first note_velocity dimension on its
    channel if there is one, otherwise from the mapping's fixed velocity
    (["note_on", channel, velocity]), otherwise 127.
    """

    def __init__(self, mapping: list):
        self.mapping = mapping
        self.sounding = {}  # dimension index -> (channel, note), 0-based channel
        self.last_note_on = {}  # channel -> most recent note sent on it
        self.velocity_index = {}  # channel -> index of its note_velocity dimension
        for i, entry in enumerate(mapping):
            if entry[0] == "note_velocity":
                self.velocity_index.setdefault(entry[1] - 1, i)

    def _velocity(self, entry: list, output_values) -> int:
        index = self.velocity_index.get(entry[1] - 1)
        if index is not None and index < len(output_values):
            # velocity 0 would be a note-off
            return max(1, value_to_midi(output_values[index]))
        if len(entry) >= 3:
            return int(min(max(entry[2], 1), 127))
        return 127

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
            if entry[0] == "note_on":
                midi_value = value_to_midi(output_values[i])
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
                        "note_on",
                        channel=channel,
                        note=midi_value,
                        velocity=self._velocity(entry, output_values),
                    )
                )
                self.sounding[i] = (channel, midi_value)
                self.last_note_on[channel] = midi_value
            elif entry[0] == "control_change":
                midi_value = value_to_midi(output_values[i], *entry[3:5])
                messages.append(
                    mido.Message(
                        "control_change",
                        channel=channel,
                        control=entry[2],
                        value=midi_value,
                    )
                )
            elif entry[0] == "pitch_bend":
                messages.append(
                    mido.Message(
                        "pitchwheel",
                        channel=channel,
                        pitch=value_to_pitch_bend(output_values[i]),
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


def midi_message_to_updates(
    msg: mido.Message, input_mapping: list
) -> List[Tuple[int, float]]:
    """Takes a MIDO message and an input mapping and returns the (index, value) updates it makes.

    A message can map to several dimensions. A CC mapped with a range
    [..., min, max] is scaled back from that range to [0, 1], so the same
    message can give different values for different dimensions.
    A note-on sets note_on dimensions on its channel to note/127 and
    note_velocity dimensions on its channel to velocity/127.
    Pitch bend is scaled from its 14-bit range to [0, 1].
    Note-ons with velocity 0 are note-offs and, like other messages that
    don't change the input, raise ValueError.
    """
    if msg.type == "note_on":
        if msg.velocity == 0:
            raise ValueError("Note-ons with velocity 0 are note-offs.")
        values = {"note_on": msg.note / 127.0, "note_velocity": msg.velocity / 127.0}
        updates = [
            (i, values[entry[0]])
            for i, entry in enumerate(input_mapping)
            if entry[0] in values and entry[1] == msg.channel + 1
        ]
    elif msg.type == "control_change":
        key = ["control_change", msg.channel + 1, msg.control]
        updates = [
            (i, midi_to_value(msg.value, *entry[3:5]))
            for i, entry in enumerate(input_mapping)
            if list(entry[:3]) == key
        ]
    elif msg.type == "pitchwheel":
        updates = [
            (i, pitch_bend_to_value(msg.pitch))
            for i, entry in enumerate(input_mapping)
            if list(entry[:2]) == ["pitch_bend", msg.channel + 1]
        ]
    else:
        raise ValueError(
            f"Only note_on, control_change and pitchwheel messages can be processed, this was a {msg.type} message."
        )
    if not updates:
        raise ValueError(f"No input mapping for {msg}.")
    return updates


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
