# IMPSY Configuration Reference

IMPSY is configured through a single TOML file — `config.toml` in the project root by default, or any path passed to `--config` / `-c` on the `run` command. The same file controls the interaction loop, the MDRNN model that's loaded, and every I/O channel (MIDI, OSC, WebSocket, serial). Logs, datasets, and trained model files are not configured here — they're discovered by convention from `logs/`, `datasets/`, and `models/`.

The example configurations in [`configs/`](../configs/) are real, working setups. The rest of this document describes every key those examples use (and a few they don't).

## Table of contents

- [Quick start](#quick-start)
- [How sections become I/O channels](#how-sections-become-io-channels)
- [Top-level keys](#top-level-keys)
- [`[interaction]`](#interaction)
- [`[model]`](#model)
- [`[midi]`](#midi)
- [`[osc]`](#osc)
- [`[websocket]`](#websocket)
- [`[serial]`](#serial)
- [`[serialmidi]`](#serialmidi)
- [MIDI mapping format](#midi-mapping-format)
- [Notes & quirks](#notes--quirks)

## Quick start

The smallest config that runs end-to-end uses a model file and one I/O channel. Here's a minimal OSC-only setup, suitable for prototyping with Pure Data, Max, or SuperCollider:

```toml
title = "Minimal IMPSY"
verbose = true
log_predictions = false

[interaction]
mode = "callresponse"
threshold = 0.1
input_thru = false

[model]
dimension = 9
file = "models/musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
size = "s"
sigmatemp = 0.01
pitemp = 1
timescale = 1

[osc]
server_ip = "0.0.0.0"
server_port = 6000
client_ip = "127.0.0.1"
client_port = 6001
```

The web UI (`poetry run ./start_impsy.py webui`) can also generate a starter config via the **Configure → Setup wizard**.

## How sections become I/O channels

The presence of a top-level table activates the matching I/O server — there's no `enabled = true` flag. Adding `[osc]` turns OSC on, removing it turns it off, and so on. All channels can be active simultaneously, and the interaction loop will read input from all of them and send predictions out through all of them.

| Section | Activates | Direction |
|---|---|---|
| `[midi]` | `MIDIServer` | bidirectional |
| `[osc]` | `OSCServer` | bidirectional |
| `[websocket]` | `WebSocketServer` | bidirectional (server side) |
| `[serial]` | `SerialServer` (CSV over serial) | bidirectional |
| `[serialmidi]` | `SerialMIDIServer` (MIDI over serial, e.g. RPi GPIO) | bidirectional |

If you don't want a channel, leave its section out. `useronly` mode (see `[interaction].mode`) is fine with no I/O channels at all if you only want to log a controller.

## Top-level keys

These sit outside any TOML table.

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `title` | string | no | — | Free-form label, e.g. `"Roland S-1 to S-1"`. Written by the web UI's setup wizard but not currently read anywhere — keep it as a note-to-self. |
| `owner` | string | no | — | Free-form metadata, e.g. your name. Not read by the runtime. |
| `description` | string | no | — | Free-form metadata. Not read by the runtime. |
| `verbose` | bool | **yes** | — | When `true`, the interaction loop prints every input/output array to stdout. Helpful for debugging mappings; noisy in normal use. |
| `log_predictions` | bool | **yes** | — | When `true`, the RNN's predictions are appended to the interaction log alongside user input. When `false` (the usual setting), only `interface` rows are logged so datasets aren't trained on the model's own output. |
| `log_input` | bool | no | — | Reserved — present in most example configs but not currently read by the runtime. Safe to leave as `true`. |

## `[interaction]`

Controls the prediction loop's behaviour.

| Key | Type | Required | Notes |
|---|---|---|---|
| `mode` | string | yes | One of `"callresponse"`, `"polyphony"`, `"battle"`, `"useronly"`. See below. |
| `threshold` | float | yes | Seconds of user inactivity before `callresponse` mode hands over to the RNN. `0.1` is a typical value. Ignored in other modes. |
| `input_thru` | bool | yes | When `true`, every user input is also sent straight back to the outputs (useful when input and output devices are different — the input device gets to "monitor itself"). When `false`, inputs only feed the RNN. |

### Modes

- **`callresponse`** — user and RNN take turns. The RNN waits until the user has been silent for `threshold` seconds, then plays back. As soon as the user moves again, the RNN stops. The most musical default for solo use.
- **`polyphony`** — user and RNN play simultaneously. RNN predictions are conditioned on user input but not on its own previous output.
- **`battle`** — RNN plays autonomously, conditioned on its own output. User input is logged but does not steer the model. Useful for autonomous performance.
- **`useronly`** — RNN does not play. Used for collecting training data without a model loaded, or for testing a controller mapping.

Mode can also be changed live over OSC (`/impsy/mode <s>`) or from the web UI's Commands page.

## `[model]`

Selects the MDRNN architecture and its temperature controls.

| Key | Type | Required | Notes |
|---|---|---|---|
| `dimension` | int | yes | Total number of values per data row, **including the inter-event time delta**. So `dimension = 9` means 8 musical parameters plus `dt`. The number of entries in each MIDI/WebSocket mapping list must equal `dimension - 1`. |
| `file` | string | yes | Path to the model file, relative to the project root. `.tflite` is preferred for inference (~20× faster than Keras); `.keras` and `.h5` also work. An empty string or missing file falls back to a `DummyMDRNN` that emits random samples — useful for `useronly` mode or testing the I/O without training. |
| `size` | string | yes | One of `"xxs"`, `"xs"`, `"s"`, `"m"`, `"l"`, `"xl"`. Maps to LSTM unit count: 16 / 32 / 64 / 128 / 256 / 512. The number of mixtures is always 5; layers are 2 except `xl` which is 3. Must match the model file's architecture — wrong size + wrong file will fail to load. |
| `sigmatemp` | float | yes | Sigma temperature applied to the Gaussian components at sampling time. Lower → predictions hug the means (smoother, more conservative). `0.01` is the usual default. |
| `pitemp` | float | yes | Pi temperature applied to the mixture weights. Lower → the model is more decisive about which Gaussian to draw from; higher → more variety between samples. `1` is neutral. |
| `timescale` | float | yes | Multiplier on the predicted `dt` before playback. `1` plays predictions at their trained tempo; `2` halves the speed; `0.5` doubles it. |

Filenames in `models/` follow the pattern `musicMDRNN-dim{N}-layers{L}-units{U}-mixtures{M}-scale{S}.{ext}` — a quick way to pick a matching `dimension`/`size` pair.

## `[midi]`

The MIDI server, using `mido` + `python-rtmidi`. Devices are referenced by name; partial matches are supported (so `"S-1"` matches `"S-1 MIDI 1"`).

| Key | Type | Required | Notes |
|---|---|---|---|
| `in_device` | array of strings | yes (if `[midi]` present) | List of MIDI input port names to open. |
| `out_device` | array of strings | yes (if `[midi]` present) | List of MIDI output port names to open. |
| `input."<port>"` | array of arrays | yes per input device | The mapping that decodes raw MIDI from this port into the model's input vector. Length must equal `dimension - 1`. See [MIDI mapping format](#midi-mapping-format). |
| `output."<port>"` | array of arrays | yes per output device | The mapping used to encode the model's output vector as MIDI for this port. Length must equal `dimension - 1`. |
| `feedback_protection` | bool | no | Default `false`. When `true`, MIDI inputs are dropped if they arrive within `feedback_threshold` of the last output AND match the most recent output note — useful when an output device echoes its own MIDI back over the same cable (e.g. Roland S-1). |
| `feedback_threshold` | float | no | Default `0.02` (20 ms). Window during which feedback protection treats incoming notes as echoes. |
| `thru_matrix."<input port>"` | array of strings | no | Optional MIDI thru routing. Each input port can list output ports that should receive its messages directly (bypassing the model). Lets you wire a controller to a synth while still feeding both into IMPSY. See [`configs/roland-s-1-quneo.toml`](../configs/roland-s-1-quneo.toml) for an example. |

Per-port mappings (`input."<name>"`, `output."<name>"`, `thru_matrix."<name>"`) use TOML's quoted-key syntax, which lets device names contain spaces and punctuation.

## `[osc]`

Bidirectional OSC, using `python-osc`. Used for talking to Pd/Max/SC, or for the web UI's Commands page.

| Key | Type | Required | Notes |
|---|---|---|---|
| `server_ip` | string | yes | The address IMPSY *listens on*. `"0.0.0.0"` accepts on all interfaces (recommended). |
| `server_port` | int | yes | Port IMPSY listens on. |
| `client_ip` | string | yes | Address to *send* predictions to. Use `"127.0.0.1"` (or `"localhost"`) for the same machine, the LAN address of the target device, or `"host.docker.internal"` from inside a Docker container. |
| `client_port` | int | yes | Port on the client to send to. |

### OSC addresses

IMPSY listens on these OSC addresses (not configurable):

| Address | Type tag | Purpose |
|---|---|---|
| `/interface` | `f f ...` | Inbound interaction values. The number of floats must equal `dimension - 1`; values are expected in `[0, 1]`. |
| `/temperature` | `f f` | Live update of `sigmatemp` and `pitemp`. |
| `/timescale` | `f` | Live update of `timescale`. |
| `/impsy/mode` | `s` | Switch interaction mode (`callresponse` / `polyphony` / `battle` / `useronly`). |
| `/impsy/reset` | `i` | Clear the LSTM hidden state. Argument is unused. |
| `/impsy/pause` | `i` | `1` to pause prediction, `0` to resume. |

Outbound predictions are sent as `/impsy` followed by `dimension - 1` floats.

> **Note on macOS:** the web UI sends commands to the OSC server. If `server_ip = "0.0.0.0"`, the web UI rewrites the destination to `127.0.0.1` because macOS rejects sends to `0.0.0.0`. This is transparent — no config change needed.

## `[websocket]`

A WebSocket server (`websockets` library) that exchanges MIDI-shaped messages with browser clients.

| Key | Type | Required | Notes |
|---|---|---|---|
| `server_ip` | string | yes | The address the WebSocket server binds to. `"0.0.0.0"` for all interfaces. |
| `server_port` | int | yes | Port the WebSocket server listens on. |
| `input` | array of arrays | yes | Mapping for incoming WebSocket messages, same format as [MIDI mappings](#midi-mapping-format). Length = `dimension - 1`. |
| `output` | array of arrays | yes | Mapping for outgoing WebSocket messages, same format as MIDI. Length = `dimension - 1`. |

Messages are slash-separated strings. Outgoing format: `/channel/{ch}/noteon/{note}/{velocity}`, `/channel/{ch}/noteoff/{note}/{velocity}`, `/channel/{ch}/cc/{ctrl}/{value}`. Incoming uses the same shape.

## `[serial]`

Plain CSV-over-serial. Each line is a comma-separated list of floats matching `dimension - 1`.

| Key | Type | Required | Notes |
|---|---|---|---|
| `port` | string | yes | OS device path (`/dev/ttyAMA0` on RPi GPIO, `/dev/tty.usbmodemXXXX` on macOS, `COM3` on Windows). |
| `baudrate` | int | yes | Serial baud rate. `115200` is typical for USB CDC; serial-MIDI uses `31250` but that's hard-coded in `[serialmidi]` (see below). |

This channel is fine for talking to micro:bit / Arduino / Pico interfaces over USB serial.

## `[serialmidi]`

Raw MIDI bytes over serial — for Raspberry Pi UART → DIN-MIDI breakouts.

| Key | Type | Required | Notes |
|---|---|---|---|
| `input` | array of arrays | yes | MIDI mapping for the serial-MIDI input. Length = `dimension - 1`. |
| `output` | array of arrays | yes | MIDI mapping for the serial-MIDI output. Length = `dimension - 1`. |

The serial port itself is taken from `[serial].port` (not from `[serialmidi]`), and the baud rate is fixed at `31250`. This means `[serialmidi]` cannot be used in isolation — `[serial]` must also be present so the port path can be resolved.

> **Known TODO:** because the runtime activates both servers whenever their tables exist, having `[serial]` and `[serialmidi]` together opens *two* readers against the same port. They will fight. A clean serial-MIDI-only path is missing today; the safest route is to use `[serial]` (CSV) or `[serialmidi]` (MIDI) but not both, and accept that "MIDI-only" still requires a stub `[serial]` table you ignore.

## MIDI mapping format

`[midi]`, `[websocket]`, and `[serialmidi]` all use the same mapping shape: an ordered list of MIDI message descriptors, one entry per dimension (excluding the time delta). The runtime asserts that `len(mapping) == dimension - 1`.

Each entry is either:

- `["note_on", channel]` — a MIDI note message on `channel` (1–16). The dimension's value (`0.0`–`1.0`) is scaled to MIDI note 0–127 for the note number; velocity is fixed at 127.
- `["control_change", channel, controller]` — a control-change message on `channel`, controller `0`–`127`. The dimension's value is scaled to `0`–`127`.
- `["control_change", channel, controller, min, max]` — same, but the value is mapped into the range `[min, max]` instead of `[0, 127]`. Useful for synth parameters that respond to a narrower CC range.

The same mapping is used in both directions. Inbound: a matching MIDI message is decoded into `(index, value/127)` and dropped into the model's input vector at `index`. Outbound: the model's output vector is encoded back into MIDI messages following the mapping.

Channels are 1-based in the config (matching standard MIDI conventions); they're decremented internally to mido's 0-based channel.

## Notes & quirks

- **`dimension` must match your data and your mappings.** The dataset that produced the model has `N` columns; the model's `dimension` must equal `N`; the number of entries in every `input.*`/`output.*` list must equal `dimension - 1`. The `*-{N}d-mdrnn.log` log filename and the `dim{N}` portion of the model filename both encode the same `N`.

- **`size` must match the model file.** If you load `musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite`, set `size = "s"` (64 units, 2 layers). Mismatches cause loading errors.

- **An empty `model.file` is valid.** It loads a `DummyMDRNN` that returns random samples. Useful for testing I/O wiring without a trained model, or for `useronly` mode.

- **Live OSC updates are partially wired.** `/temperature` and `/timescale` print but don't currently update the live model state — those handlers have `TODO` markers. Mode/reset/pause are fully wired.

- **Saving from the web UI overwrites the file verbatim.** The Configure page is a plain text editor; it does not validate TOML beyond what `tomllib` does on the next load. Keep a copy in `configs/` if you've spent time on a mapping.
