"""impsy.conformance: generate and check IMPSY conformance test vectors.

IMPSY (this package) is the reference implementation of the IMPSY data
representation. Other front ends (impsy-auv3, impsy-web, ...) re-implement
the mapping from MIDI/WebSocket messages to model input vectors, the mapping
from model output vectors back to messages, and the log -> dataset format.
This module records what the reference implementation does as JSON test
vectors (in ``spec/vectors/``) that any implementation can run in its own
test suite.

Every expected value here is produced by running the real IMPSY code
(MIDIServer, WebSocketServer, InteractionServer, dataset) against fake
ports, fake websocket clients and a scripted clock, so the vectors describe
behaviour that ships, not a separate model of it.

Regenerate the committed vectors after an intentional behaviour change:

    poetry run python -m impsy.conformance --out spec/vectors

and check they're current with ``--check``.
"""

import copy
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import click
import mido
import numpy as np

from impsy import dataset, impsio, interaction

# Bump the minor version when cases are added, the major version when the
# expected behaviour of an existing case changes.
SPEC_VERSION = "0.1.0"

DEFAULT_VECTOR_DIR = Path(__file__).resolve().parent.parent / "spec" / "vectors"

IN_PORT = "in"
OUT_PORT = "out"


# Fakes for IO endpoints


class FakeMidiInPort:
    def __init__(self, messages):
        self.messages = messages

    def iter_pending(self):
        yield from self.messages


class FakeMidiOutPort:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


class FakeWebsocketClient:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.sent = []

    def __iter__(self):
        return iter(self.messages)

    def send(self, message):
        self.sent.append(message)


class FakeClock:
    """Stands in for the ``time`` module inside impsy.interaction."""

    def __init__(self, now=0.0):
        self.now = now

    def time(self):
        return self.now


class ListLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


# Helpers


def _midi_config(dimension, input_mapping=None, output_mapping=None):
    return {
        "verbose": False,
        "model": {"dimension": dimension},
        "interaction": {"input_thru": False},
        "midi": {
            "in_device": [IN_PORT],
            "out_device": [OUT_PORT],
            "input": {IN_PORT: input_mapping or []},
            "output": {OUT_PORT: output_mapping or []},
        },
    }


def _midi_server(config, callback):
    server = impsio.MIDIServer(config, callback, lambda values: None)
    server.midi_in_port = {IN_PORT: None}
    server.midi_out_port = {OUT_PORT: FakeMidiOutPort()}
    server.last_midi_notes = {OUT_PORT: {}}
    return server


def _floats(values):
    return [float(v) for v in values]


def _interaction_server(config, initial_values, start_time):
    """An InteractionServer with its state set up but no IO, threads or sockets."""
    server = object.__new__(interaction.InteractionServer)
    server.config = config
    server.verbose = False
    server.dimension = config["model"]["dimension"]
    server.paused = False
    server.senders = []
    server.interface_input_queue = interaction.queue.Queue()
    server.last_user_interaction_time = start_time
    server.last_user_interaction_data = np.array([0.0, *initial_values])
    server._broadcast_monitor = lambda direction, values: None
    logger = logging.getLogger(f"impsy-conformance-{id(server)}")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = ListLogHandler()
    logger.addHandler(handler)
    server.logger = logger
    return server, handler


def _parse_log_line(line):
    # timestamp,source,x_1,...,x_n -- the timestamp is wall-clock and not
    # part of the vector.
    _, source, *values = line.split(",")
    return {"source": source, "values": [float(v) for v in values]}


# Runners: one per vector file. Each takes a case and returns its "expected".


def run_midi_input_case(case):
    """Each MIDI message in -> the (index, value) it produces, or None if ignored."""
    config = _midi_config(case["dimension"], input_mapping=case["input_mapping"])
    results = []
    for message_bytes in case["messages"]:
        received = []
        server = _midi_server(
            config, lambda index, value: received.append((index, value))
        )
        server.midi_in_port[IN_PORT] = FakeMidiInPort(
            [mido.Message.from_bytes(message_bytes)]
        )
        server.handle()
        if received:
            index, value = received[0]
            results.append({"index": index, "value": float(value)})
        else:
            results.append(None)
    return results


def run_midi_output_case(case):
    """A sequence of output steps -> the MIDI bytes sent at each step.

    Values pass through InteractionServer.send_back_values, so clipping to
    [0, 1] is included. MIDIServer keeps per-channel note state across steps.
    A step of {"all_notes_off": true} is what happens on disconnect.
    """
    config = _midi_config(case["dimension"], output_mapping=case["output_mapping"])
    midi = _midi_server(config, lambda index, value: None)
    server, _ = _interaction_server(config, [0.0] * (case["dimension"] - 1), 0.0)
    server.senders = [midi]
    out_port = midi.midi_out_port[OUT_PORT]
    results = []
    for step in case["steps"]:
        out_port.sent = []
        if step.get("all_notes_off"):
            midi.send_midi_note_offs()
        else:
            server.send_back_values(np.array(step["values"]))
        results.append([msg.bytes() for msg in out_port.sent])
    return results


def run_websocket_input_case(case):
    """Each websocket string in -> the (index, value) it produces, or None."""
    config = {"verbose": False, "websocket": {"input": case["input_mapping"]}}
    config["websocket"]["output"] = []
    results = []
    for message in case["messages"]:
        received = []
        server = impsio.WebSocketServer(
            config,
            lambda index, value: received.append((index, value)),
            lambda values: None,
        )
        with patch.object(impsio.click, "secho"):
            server.websocket_handler(FakeWebsocketClient([message]))
        if received:
            index, value = received[0]
            results.append({"index": index, "value": float(value)})
        else:
            results.append(None)
    return results


def run_websocket_output_case(case):
    """A sequence of output vectors -> the websocket strings sent at each step."""
    config = {
        "verbose": False,
        "websocket": {"input": [], "output": case["output_mapping"]},
    }
    server = impsio.WebSocketServer(config, lambda i, v: None, lambda v: None)
    client = FakeWebsocketClient()
    server.ws_clients.add(client)
    results = []
    for values in case["steps"]:
        client.sent = []
        server.send(np.array(values))
        results.append(list(client.sent))
    return results


def run_pipeline_case(case):
    """Timed MIDI input -> model input vectors and log rows.

    This is the full input path: MIDIServer decodes each message, then
    InteractionServer.construct_input_list merges it into the dense vector,
    computes dt from the previous interaction, logs it, and queues it for
    the model.
    """
    dimension = case["dimension"]
    config = _midi_config(dimension, input_mapping=case["input_mapping"])
    server, log_handler = _interaction_server(
        config, case["initial_values"], case["start_time"]
    )
    midi = _midi_server(config, server.construct_input_list)
    clock = FakeClock(case["start_time"])
    with patch.object(interaction, "time", clock):
        for event in case["events"]:
            clock.now = event["time"]
            midi.midi_in_port[IN_PORT] = FakeMidiInPort(
                [mido.Message.from_bytes(event["bytes"])]
            )
            midi.handle()
    model_inputs = []
    while not server.interface_input_queue.empty():
        model_inputs.append(_floats(server.interface_input_queue.get_nowait()))
    return {
        "model_inputs": model_inputs,
        "log": [_parse_log_line(line) for line in log_handler.records],
    }


def run_dataset_case(case):
    """Log file lines -> dataset rows of [dt, x_1, ..., x_n]."""
    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / f"conformance-{case['dimension']}d-mdrnn.log"
        log_file.write_text("\n".join(case["log_lines"]) + "\n")
        rows = dataset.transform_log_to_sequence_example(
            str(log_file), case["dimension"]
        )
    return [_floats(row) for row in rows]


# Cases. Inputs only; expected outputs are filled in by running the runners.

NOTE = 0x90
NOTE_OFF = 0x80
CC = 0xB0
PITCH_BEND = 0xE0


def note(n):
    """An output value that encodes to MIDI note n under both ceil and round."""
    return (n - 0.3) / 127


MIXED_MAPPING = [
    ["note_on", 1],
    ["control_change", 1, 42],
    ["note_on", 2],
    ["control_change", 16, 7],
]

MIDI_INPUT_CASES = [
    {
        "name": "note_on_value_is_pitch",
        "description": "A note-on on a mapped channel sets that dimension to note/127. Velocity is ignored.",
        "dimension": 5,
        "input_mapping": MIXED_MAPPING,
        "messages": [
            [NOTE | 0, 60, 100],
            [NOTE | 0, 60, 1],
            [NOTE | 0, 0, 127],
            [NOTE | 0, 127, 127],
            [NOTE | 1, 64, 90],
        ],
    },
    {
        "name": "note_on_velocity_zero",
        "description": "A note-on with velocity 0 (a note-off by MIDI convention) is currently treated as a new note.",
        "open_decisions": ["https://github.com/cpmpercussion/impsy/issues/99"],
        "dimension": 5,
        "input_mapping": MIXED_MAPPING,
        "messages": [[NOTE | 0, 60, 0], [NOTE | 1, 72, 0]],
    },
    {
        "name": "note_off_ignored",
        "description": "Note-off messages never change the input vector.",
        "dimension": 5,
        "input_mapping": MIXED_MAPPING,
        "messages": [[NOTE_OFF | 0, 60, 0], [NOTE_OFF | 1, 64, 64]],
    },
    {
        "name": "control_change_value",
        "description": "A CC on a mapped channel and controller sets that dimension to value/127.",
        "dimension": 5,
        "input_mapping": MIXED_MAPPING,
        "messages": [
            [CC | 0, 42, 0],
            [CC | 0, 42, 64],
            [CC | 0, 42, 127],
            [CC | 15, 7, 100],
        ],
    },
    {
        "name": "unmapped_messages_ignored",
        "description": "Messages on unmapped channels or controllers, and unsupported message types, are ignored.",
        "dimension": 5,
        "input_mapping": MIXED_MAPPING,
        "messages": [
            [NOTE | 2, 60, 100],
            [CC | 0, 43, 64],
            [CC | 1, 42, 64],
            [PITCH_BEND | 0, 0, 64],
            [0xF8],
        ],
    },
    {
        "name": "channels_are_one_based_in_config",
        "description": "Config channel 1 is status nibble 0; config channel 16 is status nibble 15.",
        "dimension": 3,
        "input_mapping": [["note_on", 16], ["control_change", 1, 1]],
        "messages": [
            [NOTE | 15, 60, 100],
            [NOTE | 0, 60, 100],
            [CC | 0, 1, 50],
            [CC | 15, 1, 50],
        ],
    },
    {
        "name": "duplicate_note_mapping_first_wins",
        "description": "When two dimensions map to note_on on the same channel, input goes to the first.",
        "open_decisions": ["https://github.com/cpmpercussion/impsy/issues/102"],
        "dimension": 3,
        "input_mapping": [["note_on", 1], ["note_on", 1]],
        "messages": [[NOTE | 0, 60, 100]],
    },
]

MIDI_OUTPUT_CASES = [
    {
        "name": "note_and_cc_encoding",
        "description": "Note dimensions become note-on with note = ceil(value * 127) and velocity 127; CC dimensions become value = ceil(value * 127). The third step (0.3 -> 38.1, 0.6 -> 76.2) is where ceil and round-to-nearest differ.",
        "open_decisions": ["https://github.com/cpmpercussion/impsy/issues/100"],
        "dimension": 5,
        "output_mapping": MIXED_MAPPING,
        "steps": [
            {"values": [0.5, 0.5, 0.0, 1.0]},
            {"values": [note(60), 0.25, 0.1, 0.999]},
            {"values": [0.3, 0.3, 0.6, 0.6]},
        ],
    },
    {
        "name": "monophonic_note_offs",
        "description": "Before each note-on, a note-off (velocity 0) is sent for the previous note on that channel, even if the note is the same.",
        "dimension": 3,
        "output_mapping": [["note_on", 1], ["note_on", 2]],
        "steps": [
            {"values": [note(60), note(64)]},
            {"values": [note(62), note(64)]},
            {"values": [0.0, 1.0]},
        ],
    },
    {
        "name": "all_notes_off",
        "description": "On disconnect, a note-off is sent for the last note on each note channel that has played.",
        "dimension": 3,
        "output_mapping": [["note_on", 1], ["note_on", 3]],
        "steps": [
            {"all_notes_off": True},
            {"values": [note(60), note(72)]},
            {"all_notes_off": True},
        ],
    },
    {
        "name": "values_clipped_to_unit_range",
        "description": "Output values are clipped to [0, 1] before encoding.",
        "dimension": 3,
        "output_mapping": [["note_on", 1], ["control_change", 1, 10]],
        "steps": [{"values": [-0.3, 1.7]}, {"values": [1.2, -5.0]}],
    },
    {
        "name": "control_change_min_max",
        "description": "A 5-element CC mapping [cc, ch, ctrl, min, max] scales the 0-127 value v to ceil((max - min) * v / 127) + min.",
        "open_decisions": ["https://github.com/cpmpercussion/impsy/issues/100"],
        "dimension": 4,
        "output_mapping": [
            ["control_change", 1, 20, 0, 63],
            ["control_change", 1, 21, 64, 127],
            ["control_change", 1, 22, 10, 20],
        ],
        "steps": [
            {"values": [0.0, 0.0, 0.0]},
            {"values": [0.5, 0.5, 0.5]},
            {"values": [1.0, 1.0, 1.0]},
        ],
    },
    {
        "name": "output_boundary_values",
        "description": "Values exactly at n/127 (in float64). ceil(value * 127) gives n here, but an implementation that holds these values as float32 and multiplies in float64 gets n + 1 for many n. Round-to-nearest gives n either way.",
        "open_decisions": ["https://github.com/cpmpercussion/impsy/issues/100"],
        "dimension": 3,
        "output_mapping": [["note_on", 1], ["control_change", 1, 1]],
        "steps": [
            {"values": [9 / 127, 9 / 127]},
            {"values": [60 / 127, 60 / 127]},
            {"values": [100 / 127, 100 / 127]},
        ],
    },
    {
        "name": "duplicate_note_mapping",
        "description": "Two note dimensions on the same channel: the second note-on first turns off the first.",
        "open_decisions": ["https://github.com/cpmpercussion/impsy/issues/102"],
        "dimension": 3,
        "output_mapping": [["note_on", 1], ["note_on", 1]],
        "steps": [{"values": [note(60), note(64)]}],
    },
]

WEBSOCKET_INPUT_CASES = [
    {
        "name": "websocket_input",
        "description": "Incoming /channel/{ch}/noteon/{note}/{vel} and /channel/{ch}/cc/{ctrl}/{value}; channels are 1-based. Note-offs are ignored; velocity-0 note-ons count as notes (see the MIDI note_on_velocity_zero case).",
        "open_decisions": ["https://github.com/cpmpercussion/impsy/issues/99"],
        "input_mapping": MIXED_MAPPING,
        "messages": [
            "/channel/1/noteon/60/100",
            "/channel/2/noteon/64/0",
            "/channel/1/noteoff/60/0",
            "/channel/1/cc/42/127",
            "/channel/16/cc/7/64",
            "/channel/3/noteon/60/100",
            "/channel/1/cc/43/64",
        ],
    },
]

WEBSOCKET_OUTPUT_CASES = [
    {
        "name": "websocket_output",
        "description": "Outgoing messages use the same wire format and the same note/CC encoding as MIDI output, including note-offs before each new note on a channel.",
        "open_decisions": ["https://github.com/cpmpercussion/impsy/issues/100"],
        "output_mapping": MIXED_MAPPING,
        "steps": [
            [0.5, 0.5, 0.0, 1.0],
            [note(60), 0.25, 0.1, 0.999],
            [0.3, 0.3, 0.6, 0.6],
        ],
    },
]

PIPELINE_CASES = [
    {
        "name": "sparse_midi_to_dense_model_input",
        "description": "Each MIDI event updates one dimension of the current input vector; the others keep their last values. The model input is [dt, x_1, ..., x_n], where dt is seconds since the previous interaction (or since start_time for the first). Each event also writes an 'interface' log row with x_1..x_n. The initial vector is random in the reference implementation; here it is set explicitly.",
        "open_decisions": [
            "https://github.com/cpmpercussion/impsy/issues/99",
            "https://github.com/cpmpercussion/impsy/issues/101",
        ],
        "dimension": 5,
        "input_mapping": MIXED_MAPPING,
        "initial_values": [0.0, 0.0, 0.0, 0.0],
        "start_time": 100.0,
        "events": [
            {"time": 100.5, "bytes": [NOTE | 0, 60, 100]},
            {"time": 100.75, "bytes": [CC | 0, 42, 127]},
            {"time": 101.0, "bytes": [NOTE | 0, 60, 0]},
            {"time": 101.25, "bytes": [NOTE_OFF | 0, 60, 0]},
            {"time": 102.0, "bytes": [NOTE | 1, 72, 80]},
            {"time": 102.125, "bytes": [CC | 15, 7, 0]},
        ],
    },
    {
        "name": "ignored_messages_do_not_reset_dt",
        "description": "Ignored messages produce no model input, and dt runs from the last message that did.",
        "dimension": 3,
        "input_mapping": [["note_on", 1], ["control_change", 1, 1]],
        "initial_values": [0.5, 0.5],
        "start_time": 0.0,
        "events": [
            {"time": 1.0, "bytes": [NOTE | 0, 48, 100]},
            {"time": 1.5, "bytes": [CC | 0, 2, 100]},
            {"time": 1.75, "bytes": [0xF8]},
            {"time": 2.0, "bytes": [CC | 0, 1, 0]},
        ],
    },
]

DATASET_CASES = [
    {
        "name": "log_to_dataset",
        "description": "Log rows are 'timestamp,source,x_1,...,x_n' with ISO 8601 timestamps. Only 'interface' rows are used. Each dataset row is [dt, x_1, ..., x_n] where dt is the time since the previous interface row, so the first row is dropped.",
        "dimension": 3,
        "log_lines": [
            "2026-09-26T10:00:00.000000,interface,0.5,0.25",
            "2026-09-26T10:00:00.500000,interface,0.4724409448818898,0.25",
            "2026-09-26T10:00:00.600000,rnn,0.1,0.2",
            "2026-09-26T10:00:01.250000,interface,0.4724409448818898,1.0",
            "2026-09-26T10:00:03.250000,interface,0.0,1.0",
        ],
    },
    {
        "name": "malformed_rows_skipped",
        "description": "Rows with too few values or unparseable numbers or timestamps are skipped; dt is measured between the remaining rows.",
        "dimension": 3,
        "log_lines": [
            "2026-09-26T10:00:00,interface,0.5,0.25",
            "2026-09-26T10:00:01,interface,0.5",
            "2026-09-26T10:00:02,interface,abc,0.25",
            "not-a-timestamp,interface,0.5,0.25",
            "2026-09-26T10:00:04,interface,0.75,0.25",
        ],
    },
]

VECTOR_FILES = {
    "midi_input.json": (
        "MIDI bytes in -> (index, value) for the model input vector, or null if ignored. Index is 0-based over x_1..x_n (dt excluded).",
        MIDI_INPUT_CASES,
        run_midi_input_case,
    ),
    "midi_output.json": (
        "Model output vectors (x_1..x_n, no dt) -> MIDI bytes sent at each step. Per-channel note state carries across steps within a case.",
        MIDI_OUTPUT_CASES,
        run_midi_output_case,
    ),
    "websocket_input.json": (
        "WebSocket message strings in -> (index, value), or null if ignored.",
        WEBSOCKET_INPUT_CASES,
        run_websocket_input_case,
    ),
    "websocket_output.json": (
        "Model output vectors -> WebSocket message strings sent at each step.",
        WEBSOCKET_OUTPUT_CASES,
        run_websocket_output_case,
    ),
    "pipeline.json": (
        "Timed MIDI input -> the model input vectors [dt, x_1..x_n] and 'interface' log rows it produces.",
        PIPELINE_CASES,
        run_pipeline_case,
    ),
    "dataset.json": (
        "Log file lines -> training dataset rows [dt, x_1..x_n].",
        DATASET_CASES,
        run_dataset_case,
    ),
}

RUNNERS = {name: runner for name, (_, _, runner) in VECTOR_FILES.items()}


def build_vectors():
    """Run every case through the reference implementation. Returns {filename: document}."""
    documents = {}
    for filename, (description, cases, runner) in VECTOR_FILES.items():
        built = []
        for case in cases:
            case = copy.deepcopy(case)
            case["expected"] = runner(copy.deepcopy(case))
            built.append(case)
        documents[filename] = {
            "spec_version": SPEC_VERSION,
            "description": description,
            "cases": built,
        }
    return documents


def _to_json(value, indent=0):
    """JSON with one item per line, except lists of scalars, which stay on one line."""
    pad = "  " * (indent + 1)
    if isinstance(value, dict):
        items = [
            f"{pad}{json.dumps(k)}: {_to_json(v, indent + 1)}" for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + "  " * indent + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        items = [f"{pad}{_to_json(v, indent + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + "\n" + "  " * indent + "]"
    return json.dumps(value)


def dump(document):
    return _to_json(document) + "\n"


@click.command()
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_VECTOR_DIR,
    show_default=True,
    help="Directory for the vector JSON files.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Don't write; exit non-zero if the files differ from what would be generated.",
)
def main(out: Path, check: bool):
    """Generate IMPSY conformance test vectors from the reference implementation."""
    documents = build_vectors()
    stale = []
    for filename, document in documents.items():
        path = out / filename
        text = dump(document)
        if check:
            if not path.exists() or path.read_text() != text:
                stale.append(filename)
        else:
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            click.echo(f"wrote {path}")
    if stale:
        raise click.ClickException(f"out of date: {', '.join(stale)}")


if __name__ == "__main__":
    main()
