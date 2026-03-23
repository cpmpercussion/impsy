# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMPSY (Interactive Music Prediction SYstem) is a Python system for musicians that uses LSTM + Mixture Density Networks (MDRNNs) to learn from and predict musical gestures in real-time. The 4-step workflow: configure I/O → log interaction data → train model → perform with predictions.

## Commands

**Setup:**
```bash
poetry install
```

**Run tests:**
```bash
poetry run pytest
poetry run pytest tests/test_model.py  # single test file
poetry run coverage run --source=impsy -m pytest  # with coverage
```

**Lint / format:**
```bash
poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
poetry run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
poetry run black .
```

**CLI commands:**
```bash
poetry run ./start_impsy.py run           # Run interaction/prediction loop
poetry run ./start_impsy.py dataset       # Generate dataset from logs
poetry run ./start_impsy.py train         # Train MDRNN model
poetry run ./start_impsy.py convert-tflite  # Convert .keras/.h5 → .tflite
poetry run ./start_impsy.py test-mdrnn    # Test MDRNN functionality
poetry run ./start_impsy.py webui         # Launch Flask web UI (port 4000)
```

## Architecture

### Core Data Flow
```
I/O (MIDI/OSC/Serial/WebSocket)
    → interaction.py (mode control + logging)
    → mdrnn.py (LSTM+MDN inference)
    → I/O (predictions back out)

logs/ → dataset.py → datasets/ (.npz)
datasets/ → train.py → models/ (.keras, .tflite)
```

### Key Modules

**`impsy/mdrnn.py`** — Neural network core. Three inference classes with a common abstract base (`MDRNNInferenceModel`):
- `TfliteMDRNN` — preferred for performance (20x+ faster than Keras), uses `ai-edge-litert` runtime
- `KerasMDRNN` — for `.keras`/`.h5` files
- `DummyMDRNN` — fallback, returns random samples when no model is available

`PredictiveMusicMDRNN` is used only during training. The model architecture is LSTM layers (2-3) feeding a Mixture Density Network (always 5 Gaussian mixtures). All inputs/outputs are scaled by 10 for training stability. Model size controls LSTM units: xxs=16, xs=32, s=64, m=128, l=256, xl=512.

**`impsy/impsio.py`** — I/O abstraction. `IOServer` abstract base with four implementations: `MIDIServer`, `OSCServer`, `SerialServer`, `WebSocketServer`. Each has `connect()`, `disconnect()`, `send()`, `handle()`, and fires `callback()` for sparse data or `dense_callback()` for dense data.

**`impsy/interaction.py`** — Interaction loop. Four modes: `callresponse` (RNN responds when user pauses), `polyphony` (simultaneous), `battle` (autonomous RNN), `useronly` (logging only). Manages LSTM state resets and interaction timing.

**`impsy/train.py`** — Training pipeline. Loads `.npz` dataset, builds model via `build_mdrnn_model()`, trains with EarlyStopping + ModelCheckpoint + TensorBoard callbacks. `SEQ_LEN=50`, `SEQ_STEP=1`, `SEED=2345`.

**`impsy/dataset.py`** — Log → dataset conversion. Reads `*-{dimension}d-mdrnn.log` CSV files, filters for `"interface"` source rows (excludes RNN predictions), computes inter-event time deltas (dt), outputs `.npz`.

**`impsy/utils.py`** — `SIZE_TO_PARAMETERS` dict mapping size strings to `(units, mixtures, layers)`. MIDI utilities, config loading, data generation for tests.

### Configuration

Runtime config is TOML (default: `config.toml`, examples in `configs/`). Key sections: `[interaction]` (mode, threshold, input_thru), `[model]` (dimension, file, size, sigmatemp, pitemp, timescale), `[midi]`, `[osc]`, `[serial]`, `[websocket]`.

`dimension` in config = number of musical parameters + 1 (for time). Log files and dataset files are named with dimension: `*-{dimension}d-mdrnn.log`.

### Tests

Fixtures in `tests/conftest.py` provide: synthetic log files, generated datasets, trained `xs`-size models, and converted TFLite/Keras/weights files. Tests use dimension=8, sequence_length=3, batch_size=3.

### Python & TensorFlow Compatibility

Supports Python 3.11, 3.12, and 3.13 with TensorFlow 2.20.0. TFLite inference uses `ai-edge-litert` (successor to `tensorflow-lite`). CI runs a full matrix across all three Python versions on Ubuntu, macOS, and Windows.
