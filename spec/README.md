# IMPSY conformance vectors

IMPSY's Python package is the reference implementation of how musical input becomes model input, and how model output becomes musical output. Other front ends, such as [impsy-auv3](https://github.com/cpmpercussion/impsy-auv3) and [impsy-web](https://github.com/cpmpercussion/impsy-web), re-implement those mappings. The files in `vectors/` record what the reference implementation does, as JSON test cases that any implementation can run in its own test suite.

If your implementation passes every case, it treats MIDI, WebSocket messages, log files and datasets the same way IMPSY does.

## Files

| File | Input | Expected |
|---|---|---|
| `midi_input.json` | a config `input_mapping` and a list of raw MIDI messages (byte arrays) | for each message, the `[index, value]` updates it makes to the input vector, or `null` if it is ignored |
| `midi_output.json` | a config `output_mapping` and a list of steps: output values `x_1..x_n`, or `{"all_notes_off": true}` | for each step, the list of MIDI messages sent (byte arrays, in order) |
| `websocket_input.json` | an input mapping and WebSocket message strings | as for `midi_input.json` |
| `websocket_output.json` | an output mapping and output vectors | for each step, the list of WebSocket strings sent |
| `playback.json` | a `timescale` and model outputs `[dt, x_1..x_n]` | for each output: seconds to `wait` before playing, the `output` values played, and the `next_model_input` fed back to the model |
| `pipeline.json` | an input mapping, an initial input vector, a start time, and timed MIDI events | `model_inputs`: every `[dt, x_1..x_n]` vector sent to the model; `log`: every log row as `{"source", "values"}` |
| `dataset.json` | the lines of a `*-{dimension}d-mdrnn.log` file | the dataset rows `[dt, x_1..x_n]` training uses |
| `model.json` | the fixed-weight model in `models/`, temperatures, and a sequence of `[dt, x_1..x_n]` inputs (with an optional LSTM reset) | the model's shape and tensor names, then for each step: the scaled input tensor, the raw MDN output, the mixture weights `pi`, means `mu` and sampling standard deviations `std` in IMPSY's units, and which mixture each uniform draw selects |

Conventions used across all files:

- Indices are 0-based over `x_1..x_n`. They never count `dt`, so index 0 is the first entry in the mapping. A message mapped to several entries sets all of them, as one interaction.
- Channels in mappings are 1-based, as in `config.toml`. In MIDI bytes, config channel `c` is status nibble `c - 1`.
- Output values in `midi_output.json` and `websocket_output.json` are what the interaction loop hands to the outputs. They can be outside `[0, 1]`, and clipping them is part of the expected behaviour.
- State carries across steps within a case (for example, the last note on each channel, used for note-offs) but never between cases.
- `pipeline.json` times are in seconds from an arbitrary origin. Log timestamps aren't part of the vectors, only each row's source and values.
- `model.json` paths such as `model_file` are relative to `spec/`. The model is 3-dimensional, with 2 LSTM layers of 16 units and 5 mixtures, and random weights, so it's small and easy to debug with, not musical. It checks that your code reads a `.tflite` file the way IMPSY does: input scaling, feeding each LSTM state output back to the matching input, splitting the MDN output, and applying the temperatures. Sampling itself is random and isn't compared.
- Each file has a `tolerance`: the absolute tolerance for comparing floats. Most files use `1e-9` because the values are exact in float64. `model.json` uses `1e-2` because TFLite float kernels differ slightly between CPUs.

## Using the vectors in another implementation

1. Copy `spec/vectors/` and `spec/models/` into your repository from a tagged IMPSY release, and record which one (a git submodule works too). Check `spec_version` in your test so an update is a deliberate change.
2. Write a test that loads each file, feeds every case's inputs through your code, and compares the result with `expected`. Compare integers and strings exactly, and floats within an absolute tolerance: the file's `tolerance`, or `1e-6` if your implementation works in float32 and the file's value is smaller.
3. If your implementation deliberately differs from a case, skip that case by name, with a comment that links to the discussion. Don't edit the vector.

Each case has a `description`. Cases whose behaviour was settled in a GitHub discussion link to it in `issues`. When IMPSY's behaviour changes, the vectors change and `spec_version` gets a major bump.

## Regenerating

The expected values come from running the real IMPSY code (`MIDIServer`, `WebSocketServer`, `InteractionServer`, `dataset`) with fake ports and a scripted clock. The cases and runners are in `impsy/conformance.py`.

```bash
poetry run python -m impsy.conformance          # write spec/vectors/
poetry run python -m impsy.conformance --check  # fail if the files are stale
poetry run pytest tests/test_conformance.py     # run the vectors against IMPSY
```

`tests/test_conformance.py` fails if the committed vectors don't match the generator, so a behaviour change in IMPSY can't slip past without updating them. When a change is intentional, regenerate and bump `SPEC_VERSION` in `impsy/conformance.py`: minor for new cases, major when an existing case's expected output changes.

The `.tflite` file in `models/` is committed rather than rebuilt in tests, because the converter's output can change between TensorFlow versions. Its weights come from a seeded generator, and `--build-model` rebuilds it. Only do that on purpose: it changes every expected value in `model.json`.

## Not covered yet

- OSC and serial (CSV and serial MIDI) IO, and MIDI feedback protection.
- Config validation, e.g. mapping length.
