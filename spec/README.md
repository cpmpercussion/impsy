# IMPSY conformance vectors

IMPSY's Python package is the reference implementation of how musical input becomes model input, and how model output becomes musical output. Other front ends, such as [impsy-auv3](https://github.com/cpmpercussion/impsy-auv3) and [impsy-web](https://github.com/cpmpercussion/impsy-web), re-implement those mappings. The files in `vectors/` record what the reference implementation does, as JSON test cases that any implementation can run in its own test suite.

If your implementation passes every case, it treats MIDI, WebSocket messages, log files and datasets the same way IMPSY does.

## Files

| File | Input | Expected |
|---|---|---|
| `midi_input.json` | a config `input_mapping` and a list of raw MIDI messages (byte arrays) | for each message, `{"index", "value"}` for the input vector, or `null` if it is ignored |
| `midi_output.json` | a config `output_mapping` and a list of steps: output values `x_1..x_n`, or `{"all_notes_off": true}` | for each step, the list of MIDI messages sent (byte arrays, in order) |
| `websocket_input.json` | an input mapping and WebSocket message strings | as for `midi_input.json` |
| `websocket_output.json` | an output mapping and output vectors | for each step, the list of WebSocket strings sent |
| `pipeline.json` | an input mapping, an initial input vector, a start time, and timed MIDI events | `model_inputs`: every `[dt, x_1..x_n]` vector sent to the model; `log`: every log row as `{"source", "values"}` |
| `dataset.json` | the lines of a `*-{dimension}d-mdrnn.log` file | the dataset rows `[dt, x_1..x_n]` training uses |

Conventions used across all files:

- `index` is 0-based over `x_1..x_n`. It never counts `dt`, so index 0 is the first entry in the mapping.
- Channels in mappings are 1-based, as in `config.toml`. In MIDI bytes, config channel `c` is status nibble `c - 1`.
- Output values in `midi_output.json` and `websocket_output.json` are what the interaction loop hands to the outputs. They can be outside `[0, 1]`, and clipping them is part of the expected behaviour.
- State carries across steps within a case (for example, the last note on each channel, used for note-offs) but never between cases.
- `pipeline.json` times are in seconds from an arbitrary origin. Log timestamps aren't part of the vectors, only each row's source and values.

## Using the vectors in another implementation

1. Copy `spec/vectors/` into your repository from a tagged IMPSY release, and record which one (a git submodule works too). Check `spec_version` in your test so an update is a deliberate change.
2. Write a test that loads each file, feeds every case's inputs through your code, and compares the result with `expected`. Compare integers and strings exactly, and floats with an absolute tolerance. `1e-6` suits implementations that work in float32.
3. If your implementation deliberately differs from a case, skip that case by name, with a comment that links to the discussion. Don't edit the vector.

Each case has a `description`. Cases that depend on a design question that is still open list the GitHub issues in `open_decisions`. Those cases record what IMPSY does *today*. When a decision changes that behaviour, the vector changes and `spec_version` gets a major bump.

## Regenerating

The expected values come from running the real IMPSY code (`MIDIServer`, `WebSocketServer`, `InteractionServer`, `dataset`) with fake ports and a scripted clock. The cases and runners are in `impsy/conformance.py`.

```bash
poetry run python -m impsy.conformance          # write spec/vectors/
poetry run python -m impsy.conformance --check  # fail if the files are stale
poetry run pytest tests/test_conformance.py     # run the vectors against IMPSY
```

`tests/test_conformance.py` fails if the committed vectors don't match the generator, so a behaviour change in IMPSY can't slip past without updating them. When a change is intentional, regenerate and bump `SPEC_VERSION` in `impsy/conformance.py`: minor for new cases, major when an existing case's expected output changes.

## Not covered yet

- Model inference: a small fixed-weight model with an input sequence and the expected mixture parameters before sampling.
- The model-output playback path: the `dt` floor, `timescale`, and what is fed back into the model (see [#103](https://github.com/cpmpercussion/impsy/issues/103)).
- OSC and serial (CSV and serial MIDI) IO, and MIDI feedback protection.
- Config validation, e.g. mapping length and duplicate entries ([#102](https://github.com/cpmpercussion/impsy/issues/102)).
