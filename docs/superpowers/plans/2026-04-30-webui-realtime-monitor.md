# Webui Realtime Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/realtime` page in the webui that displays live in/out channel values from the running IMPSY interaction server as bar indicators, plus fix the bundled `osc.client_ip` wildcard bug.

**Architecture:** IMPSY broadcasts every input and output vector to a localhost-only OSC monitor channel (port from `[webui].monitor_port`, default `4001`). The webui owns a lazily-started OSC listener that caches the latest values; a polling browser page reads them via JSON. User-configured `[osc]` routing is untouched.

**Tech Stack:** Python 3.11+, Flask, python-osc, Bootstrap 5 (already in `base.html`), vanilla JS (no new front-end deps).

**Spec:** [`docs/superpowers/specs/2026-04-30-webui-realtime-monitor-design.md`](../specs/2026-04-30-webui-realtime-monitor-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `impsy/impsio.py` | Modify (`OSCServer.__init__`) | Coerce wildcard `client_ip` to `127.0.0.1`. |
| `impsy/interaction.py` | Modify | Build `_monitor_client` in `__init__`; broadcast on `dense_callback`/`construct_input_list`/`send_back_values`. |
| `impsy/web_interface.py` | Modify | Add `MonitorListener`, `compute_channel_labels`, `/realtime` and `/realtime/data` routes; update `setup_config` wizard. |
| `impsy/templates/base.html` | Modify | Add Realtime nav entry. |
| `impsy/templates/realtime.html` | Create | The page (extends `base.html`), Bootstrap progress bars + ~30 lines of JS. |
| `tests/test_impsio.py` | Modify | Wildcard coercion test. |
| `tests/test_interaction.py` | Modify | Hook-firing tests, error-swallow test, fixture updates. |
| `tests/test_webui.py` | Modify | `MonitorListener` round-trip, `/realtime/data` shape, label generation, page smoke test. |
| `docs/config.md` | Modify | New `[webui]` section; OSC addresses table entries for `/monitor/{in,out}`. |
| `configs/default.toml` | Modify | Add `[webui] monitor_port = 4001` so example matches docs. |

---

## Task 1: Coerce wildcard `osc.client_ip` to localhost

**Files:**
- Modify: `impsy/impsio.py` (`OSCServer.__init__`, around lines 380–405)
- Test: `tests/test_impsio.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_impsio.py` (after the existing OSC tests, before `# Serial handler tests`):

```python
def test_osc_client_ip_wildcard_coerced_to_localhost(io_config, sparse_callback, dense_callback):
    """OSCServer must coerce wildcard client_ip to 127.0.0.1 — sending UDP to
    0.0.0.0 raises 'No route to host' on macOS.
    """
    io_config["osc"]["client_ip"] = "0.0.0.0"
    with patch("impsy.impsio.osc_server.ThreadingOSCUDPServer"), patch(
        "impsy.impsio.udp_client.SimpleUDPClient"
    ) as mock_client:
        impsio.OSCServer(io_config, sparse_callback, dense_callback)
    mock_client.assert_called_with("127.0.0.1", io_config["osc"]["client_port"])


def test_osc_client_ip_concrete_address_passed_through(
    io_config, sparse_callback, dense_callback
):
    """Concrete (non-wildcard) client_ip values must be used as-is."""
    io_config["osc"]["client_ip"] = "192.168.1.50"
    with patch("impsy.impsio.osc_server.ThreadingOSCUDPServer"), patch(
        "impsy.impsio.udp_client.SimpleUDPClient"
    ) as mock_client:
        impsio.OSCServer(io_config, sparse_callback, dense_callback)
    mock_client.assert_called_with("192.168.1.50", io_config["osc"]["client_port"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_impsio.py::test_osc_client_ip_wildcard_coerced_to_localhost -v
```

Expected: FAIL — `mock_client.assert_called_with("127.0.0.1", ...)` fails because current code passes `"0.0.0.0"` through verbatim.

- [ ] **Step 3: Implement the coercion**

In `impsy/impsio.py`, replace the existing `self.osc_client = ...` block in `OSCServer.__init__` (around line 387–389):

```python
        client_ip = config["osc"]["client_ip"]
        if client_ip in ("0.0.0.0", "::", ""):
            click.secho(
                f"OSC: client_ip {client_ip!r} is not a valid send destination; "
                "coercing to 127.0.0.1.",
                fg="yellow",
            )
            client_ip = "127.0.0.1"
        self.osc_client = udp_client.SimpleUDPClient(
            client_ip, config["osc"]["client_port"]
        )
```

- [ ] **Step 4: Run tests to verify both pass**

```bash
poetry run pytest tests/test_impsio.py -v
```

Expected: all pass, including the two new tests.

- [ ] **Step 5: Commit**

```bash
git add impsy/impsio.py tests/test_impsio.py
git commit -m "Coerce wildcard osc.client_ip to localhost on IMPSY side"
```

---

## Task 2: Add monitor broadcaster to InteractionServer

**Files:**
- Modify: `impsy/interaction.py` (top imports + `__init__` + new method)
- Test: `tests/test_interaction.py`

- [ ] **Step 1: Add the test for `_broadcast_monitor`**

Add to `tests/test_interaction.py` (after `test_handle_command_temp_before_network_loaded`):

```python
def test_broadcast_monitor_sends_in(interaction_server, monkeypatch):
    """_broadcast_monitor with direction='in' should send /monitor/in."""
    sent = []
    monkeypatch.setattr(
        interaction_server._monitor_client,
        "send_message",
        lambda addr, vals: sent.append((addr, list(vals))),
    )
    interaction_server._broadcast_monitor("in", [0.1, 0.2, 0.3])
    assert sent == [("/monitor/in", [0.1, 0.2, 0.3])]


def test_broadcast_monitor_sends_out(interaction_server, monkeypatch):
    """_broadcast_monitor with direction='out' should send /monitor/out."""
    sent = []
    monkeypatch.setattr(
        interaction_server._monitor_client,
        "send_message",
        lambda addr, vals: sent.append((addr, list(vals))),
    )
    interaction_server._broadcast_monitor("out", [0.4, 0.5])
    assert sent == [("/monitor/out", [0.4, 0.5])]


def test_broadcast_monitor_swallows_errors(interaction_server, monkeypatch):
    """A failing send (no listener) must not propagate."""
    def boom(addr, vals):
        raise OSError("no listener")

    monkeypatch.setattr(interaction_server._monitor_client, "send_message", boom)
    # Must not raise:
    interaction_server._broadcast_monitor("in", [0.1])


def test_monitor_port_defaults_to_4001(interaction_server):
    """When [webui] is missing the monitor port falls back to 4001."""
    assert interaction_server._monitor_port == 4001


def test_monitor_port_read_from_config(default_config, log_location):
    """When [webui].monitor_port is set the InteractionServer uses it."""
    cfg = copy.deepcopy(default_config)
    cfg["webui"] = {"monitor_port": 4042}
    server = interaction.InteractionServer(cfg, log_location=log_location)
    try:
        assert server._monitor_port == 4042
    finally:
        server.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_interaction.py -k broadcast_monitor -v
poetry run pytest tests/test_interaction.py -k monitor_port -v
```

Expected: AttributeError — `_broadcast_monitor` / `_monitor_client` / `_monitor_port` don't exist yet.

- [ ] **Step 3: Add the import and the broadcaster**

At the top of `impsy/interaction.py`, add the import (after the other imports):

```python
from pythonosc import udp_client
```

In `InteractionServer.__init__`, immediately after the existing line `self._reset_requested = False` (and before `self.net = None`), add:

```python
        self._monitor_port = self.config.get("webui", {}).get("monitor_port", 4001)
        self._monitor_client = udp_client.SimpleUDPClient(
            "127.0.0.1", self._monitor_port
        )
```

Add the new method on `InteractionServer` (place it next to `handle_command`):

```python
    def _broadcast_monitor(self, direction: str, values) -> None:
        """Send a copy of the latest in/out vector to the localhost monitor port.

        Failures are swallowed: a non-running webui must never break IMPSY.
        """
        addr = "/monitor/in" if direction == "in" else "/monitor/out"
        try:
            self._monitor_client.send_message(addr, list(values))
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_interaction.py -k "broadcast_monitor or monitor_port" -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add impsy/interaction.py tests/test_interaction.py
git commit -m "Add InteractionServer._broadcast_monitor + monitor_port config"
```

---

## Task 3: Hook broadcaster into `dense_callback`

**Files:**
- Modify: `impsy/interaction.py` (`dense_callback`)
- Test: `tests/test_interaction.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_interaction.py`:

```python
def test_dense_callback_broadcasts_in(
    interaction_server, default_dimension, monkeypatch
):
    """dense_callback should fire _broadcast_monitor with direction='in'."""
    sent = []
    monkeypatch.setattr(
        interaction_server,
        "_broadcast_monitor",
        lambda d, v: sent.append((d, list(v))),
    )
    values = list(np.random.rand(default_dimension - 1))
    interaction_server.dense_callback(values)
    assert sent == [("in", values)]
    # drain the queue dense_callback populated
    while not interaction_server.interface_input_queue.empty():
        interaction_server.interface_input_queue.get_nowait()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_interaction.py::test_dense_callback_broadcasts_in -v
```

Expected: FAIL — `sent` is empty because `dense_callback` doesn't call `_broadcast_monitor` yet.

- [ ] **Step 3: Add the hook**

In `impsy/interaction.py`, in `dense_callback`, immediately after `values_arr = np.array(values)`, add:

```python
        self._broadcast_monitor("in", values_arr)
```

So the method now opens with:

```python
    def dense_callback(self, values) -> None:
        """insert a dense input list into the interaction stream (e.g., when receiving OSC)."""
        values_arr = np.array(values)
        self._broadcast_monitor("in", values_arr)
        if self.verbose:
            print_io("in", values_arr, "yellow")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
poetry run pytest tests/test_interaction.py::test_dense_callback_broadcasts_in -v
```

Expected: PASS.

- [ ] **Step 5: Run the full interaction suite as a regression check**

```bash
poetry run pytest tests/test_interaction.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add impsy/interaction.py tests/test_interaction.py
git commit -m "Broadcast OSC input dense_callback values to monitor channel"
```

---

## Task 4: Hook broadcaster into `construct_input_list`

**Files:**
- Modify: `impsy/interaction.py` (`construct_input_list`)
- Test: `tests/test_interaction.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_interaction.py`:

```python
def test_construct_input_list_broadcasts_in(interaction_server, monkeypatch):
    """construct_input_list should fire _broadcast_monitor with direction='in'.

    The broadcasted vector must reflect the just-applied (index, value) update.
    """
    sent = []
    monkeypatch.setattr(
        interaction_server,
        "_broadcast_monitor",
        lambda d, v: sent.append((d, list(v))),
    )
    interaction_server.construct_input_list(0, 0.5)
    assert len(sent) == 1
    assert sent[0][0] == "in"
    # values[0] = 0.5 should appear in the broadcasted vector
    assert sent[0][1][0] == 0.5
    # drain
    while not interaction_server.interface_input_queue.empty():
        interaction_server.interface_input_queue.get_nowait()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_interaction.py::test_construct_input_list_broadcasts_in -v
```

Expected: FAIL — `sent` is empty.

- [ ] **Step 3: Add the hook**

In `impsy/interaction.py`, in `construct_input_list`, immediately after `values[index] = value`, add:

```python
        self._broadcast_monitor("in", values)
```

So the relevant lines now read:

```python
    def construct_input_list(self, index: int, value: float) -> list:
        """constructs a dense input list from a sparse format (e.g., when receiving MIDI)"""
        # set up dense interaction list
        values = self.last_user_interaction_data[1:]
        values[index] = value
        self._broadcast_monitor("in", values)
        # log
```

- [ ] **Step 4: Run test to verify it passes**

```bash
poetry run pytest tests/test_interaction.py::test_construct_input_list_broadcasts_in -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add impsy/interaction.py tests/test_interaction.py
git commit -m "Broadcast MIDI/sparse input values to monitor channel"
```

---

## Task 5: Hook broadcaster into `send_back_values`

**Files:**
- Modify: `impsy/interaction.py` (`send_back_values`)
- Test: `tests/test_interaction.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_interaction.py`:

```python
def test_send_back_values_broadcasts_out(
    interaction_server, default_dimension, monkeypatch
):
    """send_back_values should fire _broadcast_monitor with direction='out'."""
    sent = []
    monkeypatch.setattr(
        interaction_server,
        "_broadcast_monitor",
        lambda d, v: sent.append((d, list(v))),
    )
    interaction_server.paused = False
    values = np.array([0.2] * (default_dimension - 1))
    interaction_server.send_back_values(values)
    assert len(sent) == 1
    assert sent[0][0] == "out"
    assert sent[0][1] == [0.2] * (default_dimension - 1)


def test_send_back_values_skips_broadcast_when_paused(
    interaction_server, default_dimension, monkeypatch
):
    """When paused, send_back_values returns early — no broadcast."""
    sent = []
    monkeypatch.setattr(
        interaction_server,
        "_broadcast_monitor",
        lambda d, v: sent.append((d, list(v))),
    )
    interaction_server.paused = True
    interaction_server.send_back_values(np.array([0.5] * (default_dimension - 1)))
    assert sent == []
    interaction_server.paused = False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_interaction.py -k send_back_values_broadcasts -v
```

Expected: FAIL — `sent` is empty.

- [ ] **Step 3: Add the hook**

In `impsy/interaction.py`, in `send_back_values`, immediately after the clipping line `output = np.minimum(np.maximum(output_values, 0), 1)`, add:

```python
        self._broadcast_monitor("out", output)
```

So the method now reads:

```python
    def send_back_values(self, output_values):
        """sends back sound commands to the MIDI/OSC/WebSockets outputs"""
        if self.paused:
            return
        output = np.minimum(np.maximum(output_values, 0), 1)
        self._broadcast_monitor("out", output)
        if self.verbose:
            print_io("out", output, "green")
        for sender in self.senders:
            sender.send(output)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_interaction.py -k send_back_values -v
```

Expected: PASS for both new tests; existing `test_send_values` still passes.

- [ ] **Step 5: Commit**

```bash
git add impsy/interaction.py tests/test_interaction.py
git commit -m "Broadcast outgoing values to monitor channel"
```

---

## Task 6: Add `MonitorListener` to webui

**Files:**
- Modify: `impsy/web_interface.py` (top imports + new class)
- Test: `tests/test_webui.py`

- [ ] **Step 1: Write the round-trip test**

Add `import time` and `from pythonosc import udp_client` at the top of `tests/test_webui.py` (alongside the existing `import os`, `import io`).

Then append at the end of the file:

```python
def test_monitor_listener_roundtrip():
    """A real /monitor/in or /monitor/out packet updates the listener's state."""
    from impsy.web_interface import MonitorListener

    listener = MonitorListener(port=14010)
    listener.start()
    try:
        client = udp_client.SimpleUDPClient("127.0.0.1", 14010)
        client.send_message("/monitor/in", [0.1, 0.2, 0.3])
        # UDP is async — wait briefly for the receiver thread
        deadline = time.time() + 1.0
        while time.time() < deadline and listener.latest_in is None:
            time.sleep(0.02)
        assert listener.latest_in == [0.1, 0.2, 0.3]
        assert listener.in_updated_at > 0

        client.send_message("/monitor/out", [0.4, 0.5])
        deadline = time.time() + 1.0
        while time.time() < deadline and listener.latest_out is None:
            time.sleep(0.02)
        assert listener.latest_out == [0.4, 0.5]
    finally:
        listener.stop()


def test_monitor_listener_start_is_idempotent():
    """Calling start() twice must not raise (port already in use)."""
    from impsy.web_interface import MonitorListener

    listener = MonitorListener(port=14011)
    listener.start()
    try:
        listener.start()  # second call is a no-op
    finally:
        listener.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_webui.py -k monitor_listener -v
```

Expected: FAIL — `MonitorListener` doesn't exist.

- [ ] **Step 3: Add the class**

At the top of `impsy/web_interface.py` (with the other imports, after `from pathlib import Path`), add:

```python
import time
from threading import Thread
from pythonosc import dispatcher, osc_server
```

Then add the class somewhere convenient (after the existing helpers, before `# === Routes ===`):

```python
class MonitorListener:
    """Lazily-started OSC listener that mirrors IMPSY's /monitor/{in,out} stream.

    Stores the most recently received input/output vectors and timestamps so the
    /realtime page can render them. UDP packets that arrive when no listener is
    running are silently lost; that's intentional — the monitor is non-essential.
    """

    def __init__(self, port: int):
        self.port = port
        self.latest_in = None
        self.latest_out = None
        self.in_updated_at = 0.0
        self.out_updated_at = 0.0
        self._server = None
        self._thread = None

    def start(self) -> None:
        if self._server is not None:
            return  # idempotent
        disp = dispatcher.Dispatcher()
        disp.map("/monitor/in", self._on_in)
        disp.map("/monitor/out", self._on_out)
        self._server = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", self.port), disp
        )
        self._thread = Thread(
            target=self._server.serve_forever,
            name="webui_monitor_thread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        try:
            self._server.socket.close()
        except Exception:
            pass
        self._server = None
        self._thread = None

    def _on_in(self, address, *args):
        self.latest_in = list(args)
        self.in_updated_at = time.time()

    def _on_out(self, address, *args):
        self.latest_out = list(args)
        self.out_updated_at = time.time()


_monitor_listener: MonitorListener | None = None


def _ensure_monitor_listener():
    """Lazily build and start the module-level monitor listener."""
    global _monitor_listener
    if _monitor_listener is not None:
        return _monitor_listener
    port = 4001
    try:
        with open(CONFIG_FILE, "rb") as f:
            cfg = tomllib.load(f)
        port = cfg.get("webui", {}).get("monitor_port", 4001)
    except Exception:
        pass
    _monitor_listener = MonitorListener(port)
    _monitor_listener.start()
    return _monitor_listener
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_webui.py -k monitor_listener -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add impsy/web_interface.py tests/test_webui.py
git commit -m "Add MonitorListener for webui realtime monitor"
```

---

## Task 7: Add `compute_channel_labels` helper

**Files:**
- Modify: `impsy/web_interface.py`
- Test: `tests/test_webui.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_webui.py`:

```python
def test_compute_channel_labels_midi_note_and_cc():
    """MIDI note_on/control_change entries get readable labels."""
    from impsy.web_interface import compute_channel_labels

    cfg = {
        "model": {"dimension": 4},
        "midi": {
            "input": {
                "Foo": [
                    ["note_on", 1],
                    ["control_change", 2, 19],
                    ["control_change", 3, 20, 0, 127],
                ]
            }
        },
    }
    labels = compute_channel_labels(cfg)
    assert labels == ["Note ch1", "CC2:19", "CC3:20"]


def test_compute_channel_labels_picks_first_port_alphabetically():
    """Multi-port MIDI configs use the first port (sorted by name) for labels."""
    from impsy.web_interface import compute_channel_labels

    cfg = {
        "model": {"dimension": 3},
        "midi": {
            "input": {
                "Zzz Last": [["note_on", 9]],
                "Aaa First": [["note_on", 1], ["control_change", 1, 7]],
            }
        },
    }
    labels = compute_channel_labels(cfg)
    assert labels == ["Note ch1", "CC1:7"]


def test_compute_channel_labels_falls_back_to_numeric():
    """OSC-only or empty-MIDI configs return Ch 0..Ch N-1."""
    from impsy.web_interface import compute_channel_labels

    cfg = {"model": {"dimension": 5}, "osc": {"server_port": 6000}}
    labels = compute_channel_labels(cfg)
    assert labels == ["Ch 0", "Ch 1", "Ch 2", "Ch 3"]


def test_compute_channel_labels_pads_when_mapping_too_short():
    """If MIDI mapping is shorter than dimension-1, pad the rest with Ch N."""
    from impsy.web_interface import compute_channel_labels

    cfg = {
        "model": {"dimension": 5},
        "midi": {"input": {"Foo": [["note_on", 1], ["control_change", 1, 7]]}},
    }
    labels = compute_channel_labels(cfg)
    assert labels == ["Note ch1", "CC1:7", "Ch 2", "Ch 3"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_webui.py -k compute_channel_labels -v
```

Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Implement the helper**

Add to `impsy/web_interface.py` (next to other helpers like `get_osc_config`):

```python
def compute_channel_labels(config: dict) -> list[str]:
    """Return `dimension - 1` human-readable labels for the realtime sliders.

    Pulled from the first MIDI input port's mapping when [midi] is present;
    falls back to "Ch N" for any positions not covered by the mapping or when
    no MIDI config exists.
    """
    dimension = config.get("model", {}).get("dimension", 0)
    n = max(0, dimension - 1)
    midi_input = config.get("midi", {}).get("input", {})
    if isinstance(midi_input, dict) and midi_input:
        first_port = sorted(midi_input.keys())[0]
        mapping = midi_input[first_port]
    else:
        mapping = []

    labels: list[str] = []
    for i in range(n):
        if i < len(mapping):
            entry = mapping[i]
            if entry[0] == "note_on":
                labels.append(f"Note ch{entry[1]}")
            elif entry[0] == "control_change":
                labels.append(f"CC{entry[1]}:{entry[2]}")
            else:
                labels.append(f"Ch {i}")
        else:
            labels.append(f"Ch {i}")
    return labels
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_webui.py -k compute_channel_labels -v
```

Expected: PASS for all four.

- [ ] **Step 5: Commit**

```bash
git add impsy/web_interface.py tests/test_webui.py
git commit -m "Add compute_channel_labels helper for realtime page"
```

---

## Task 8: Add `/realtime/data` JSON endpoint

**Files:**
- Modify: `impsy/web_interface.py`
- Test: `tests/test_webui.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_webui.py`:

```python
def test_realtime_data_returns_listener_state(client):
    """/realtime/data returns the listener's latest values plus age in ms."""
    from impsy import web_interface

    web_interface._monitor_listener = web_interface.MonitorListener(port=14012)
    web_interface._monitor_listener.latest_in = [0.1, 0.2]
    web_interface._monitor_listener.latest_out = [0.3, 0.4]
    web_interface._monitor_listener.in_updated_at = time.time()
    web_interface._monitor_listener.out_updated_at = time.time() - 0.5
    try:
        response = client.get("/realtime/data")
        assert response.status_code == 200
        data = response.get_json()
        assert data["in"] == [0.1, 0.2]
        assert data["out"] == [0.3, 0.4]
        assert 0 <= data["in_age_ms"] < 200
        assert 400 <= data["out_age_ms"] < 800
    finally:
        web_interface._monitor_listener.stop()
        web_interface._monitor_listener = None


def test_realtime_data_returns_nulls_when_empty(client):
    """If no packets seen yet, /realtime/data returns null arrays."""
    from impsy import web_interface

    web_interface._monitor_listener = web_interface.MonitorListener(port=14013)
    try:
        response = client.get("/realtime/data")
        data = response.get_json()
        assert data["in"] is None
        assert data["out"] is None
        assert data["in_age_ms"] is None
        assert data["out_age_ms"] is None
    finally:
        web_interface._monitor_listener.stop()
        web_interface._monitor_listener = None
```

(`import time` is already at the top from Task 6.)

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_webui.py -k realtime_data -v
```

Expected: FAIL — route doesn't exist (404).

- [ ] **Step 3: Add the route**

Add to `impsy/web_interface.py` (with the other routes, near `/commands`):

```python
@app.route("/realtime/data")
def realtime_data():
    """Return the latest in/out values plus their age in ms."""
    listener = _ensure_monitor_listener()
    now = time.time()

    def age_ms(values, updated_at):
        if values is None:
            return None
        return int((now - updated_at) * 1000)

    return {
        "in": listener.latest_in,
        "out": listener.latest_out,
        "in_age_ms": age_ms(listener.latest_in, listener.in_updated_at),
        "out_age_ms": age_ms(listener.latest_out, listener.out_updated_at),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_webui.py -k realtime_data -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add impsy/web_interface.py tests/test_webui.py
git commit -m "Add /realtime/data JSON endpoint"
```

---

## Task 9: Add `/realtime` page route, template, nav entry

**Files:**
- Modify: `impsy/web_interface.py` (new route)
- Modify: `impsy/templates/base.html` (nav entry)
- Create: `impsy/templates/realtime.html`
- Test: `tests/test_webui.py`

- [ ] **Step 1: Write the smoke test**

Add to `tests/test_webui.py`:

```python
def test_realtime_route_renders(client):
    """/realtime returns 200 and includes the channel label table."""
    from impsy import web_interface

    web_interface._monitor_listener = web_interface.MonitorListener(port=14014)
    try:
        response = client.get("/realtime")
        assert response.status_code == 200
        assert b"Realtime" in response.data
        # the page should mention the monitor port it's listening on
        assert b"14014" in response.data or b"4001" in response.data
        # at least one channel row
        assert b"progress-bar" in response.data
    finally:
        web_interface._monitor_listener.stop()
        web_interface._monitor_listener = None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_webui.py::test_realtime_route_renders -v
```

Expected: FAIL — route doesn't exist.

- [ ] **Step 3: Create the template**

Create `impsy/templates/realtime.html`:

```html
{% extends "base.html" %}
{% block title %}IMPSY - Realtime{% endblock %}
{% block content %}
<h1 class="mb-2">Realtime Monitor</h1>
<p class="text-secondary mb-4">
  Live values flowing through IMPSY. Cyan = inputs (incoming from controllers),
  green = outputs (predictions and thru). Listening on UDP port
  <code>{{ monitor_port }}</code>.
</p>

<div id="status" class="alert alert-info">Waiting for data&hellip;</div>

<table class="table table-sm align-middle" id="channels">
  <thead>
    <tr>
      <th style="width: 12rem;">Channel</th>
      <th>In</th>
      <th>Out</th>
      <th class="text-end" style="width: 9rem;">Values</th>
    </tr>
  </thead>
  <tbody>
    {% for label in labels %}
    <tr data-channel="{{ loop.index0 }}">
      <td><code>{{ label }}</code></td>
      <td>
        <div class="progress" style="height: 1rem;">
          <div class="progress-bar bg-info bar-in" style="width: 0%"></div>
        </div>
      </td>
      <td>
        <div class="progress" style="height: 1rem;">
          <div class="progress-bar bg-success bar-out" style="width: 0%"></div>
        </div>
      </td>
      <td class="text-end font-monospace small">
        <span class="val-in">&mdash;</span> /
        <span class="val-out">&mdash;</span>
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}

{% block scripts %}
<script>
async function poll() {
  try {
    const resp = await fetch('/realtime/data');
    if (!resp.ok) return;
    const data = await resp.json();
    const status = document.getElementById('status');
    if (data.in === null && data.out === null) {
      status.textContent = 'Waiting for data…';
      status.className = 'alert alert-info';
      return;
    }
    const inAge = data.in_age_ms ?? Infinity;
    const outAge = data.out_age_ms ?? Infinity;
    status.textContent = `Streaming. in age ${data.in_age_ms ?? '—'}ms, out age ${data.out_age_ms ?? '—'}ms`;
    status.className = 'alert alert-success';

    document.querySelectorAll('#channels tbody tr').forEach(row => {
      const idx = parseInt(row.dataset.channel);
      const inVal = data.in ? data.in[idx] : null;
      const outVal = data.out ? data.out[idx] : null;
      row.querySelector('.bar-in').style.width = (inVal != null ? Math.max(0, Math.min(1, inVal)) * 100 : 0) + '%';
      row.querySelector('.bar-out').style.width = (outVal != null ? Math.max(0, Math.min(1, outVal)) * 100 : 0) + '%';
      row.querySelector('.val-in').textContent = inVal != null ? inVal.toFixed(3) : '—';
      row.querySelector('.val-out').textContent = outVal != null ? outVal.toFixed(3) : '—';
      row.style.opacity = (inAge > 2000 && outAge > 2000) ? 0.5 : 1.0;
    });
  } catch (e) {
    console.error('poll failed', e);
  }
}
setInterval(poll, 100);
poll();
</script>
{% endblock %}
```

- [ ] **Step 4: Add the nav entry**

In `impsy/templates/base.html`, immediately before the existing `{% if osc_enabled %}` block (Commands link), add:

```html
                    <li class="nav-item">
                        <a class="nav-link {% if active_page == 'realtime' %}active{% endif %}" href="{{ url_for('realtime') }}">Realtime</a>
                    </li>
```

- [ ] **Step 5: Add the route**

In `impsy/web_interface.py`, near `/realtime/data`:

```python
@app.route("/realtime")
def realtime():
    """Live in/out monitoring page."""
    listener = _ensure_monitor_listener()
    cfg = {}
    try:
        with open(CONFIG_FILE, "rb") as f:
            cfg = tomllib.load(f)
    except Exception:
        pass
    labels = compute_channel_labels(cfg)
    return render_template(
        "realtime.html",
        active_page="realtime",
        monitor_port=listener.port,
        labels=labels,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
poetry run pytest tests/test_webui.py -k realtime -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add impsy/web_interface.py impsy/templates/realtime.html impsy/templates/base.html tests/test_webui.py
git commit -m "Add /realtime page with bar indicators and polling"
```

---

## Task 10: Update `setup_config` wizard to write `[webui].monitor_port`

**Files:**
- Modify: `impsy/web_interface.py` (`setup_config`)

- [ ] **Step 1: Find the wizard's config-line builder**

Open `impsy/web_interface.py` and locate `setup_config` (around line 380). It builds a `config_lines` list and appends I/O sections conditionally.

- [ ] **Step 2: Add the [webui] section unconditionally**

Just before the `config_text = "\n".join(config_lines) + "\n"` line, add:

```python
        config_lines += [
            "",
            "[webui]",
            "monitor_port = 4001  # Localhost port the webui's Realtime page listens on.",
        ]
```

- [ ] **Step 3: Update the default example config**

In `configs/default.toml`, append at the end:

```toml

[webui]
monitor_port = 4001  # Localhost port the webui's Realtime page listens on.
```

- [ ] **Step 4: Smoke-check the wizard output**

```bash
poetry run python -c "
from impsy.web_interface import app
client = app.test_client()
client.post('/config/setup', data={
    'title': 'Wizard Test', 'mode': 'callresponse', 'dimension': 4,
    'model_size': 's', 'io_osc': 'on'
})
print(open('config.toml').read())
" | grep -A1 webui
```

Expected output includes:

```
[webui]
monitor_port = 4001  # Localhost port the webui's Realtime page listens on.
```

(If you ran this in a clean checkout, restore your local `config.toml` afterwards.)

- [ ] **Step 5: Run the test suite**

```bash
poetry run pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add impsy/web_interface.py configs/default.toml
git commit -m "Wire [webui].monitor_port into setup wizard and default config"
```

---

## Task 11: Update `docs/config.md` with `[webui]` section

**Files:**
- Modify: `docs/config.md`

- [ ] **Step 1: Add the [webui] section**

In `docs/config.md`, immediately after the `## `[serialmidi]`` section and before `## MIDI mapping format`, insert:

```markdown
## `[webui]`

Optional. Settings used by the webui's `/realtime` page when it monitors a running interaction server.

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `monitor_port` | int | no | `4001` | Localhost UDP port. IMPSY broadcasts every input and output vector here as `/monitor/in <f f ...>` and `/monitor/out <f f ...>`; the webui's `/realtime` page binds to the same port to display them. Sender and receiver are always `127.0.0.1`. If the section is omitted both sides default to `4001`, so existing configs work without edits. |

```

- [ ] **Step 2: Add the monitor addresses to the OSC table**

In the existing OSC addresses table in the `[osc]` section, add two rows at the bottom:

```markdown
| `/monitor/in` | `f f ...` | Internal — emitted by IMPSY on `127.0.0.1:[webui].monitor_port` whenever an input vector lands. Consumed by the webui's `/realtime` page. |
| `/monitor/out` | `f f ...` | Internal — emitted by IMPSY before sending output values to configured I/O. Consumed by the webui's `/realtime` page. |
```

- [ ] **Step 3: Commit**

```bash
git add docs/config.md
git commit -m "Document [webui] section and /monitor OSC addresses"
```

---

## Task 12: Final integration check

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full test suite**

```bash
poetry run pytest -q
```

Expected: all pass (103 baseline + 2 from Task 1 + 5 from Task 2 + 1 from Task 3 + 1 from Task 4 + 2 from Task 5 + 2 from Task 6 + 4 from Task 7 + 2 from Task 8 + 1 from Task 9 = 123 passing).

- [ ] **Step 2: Manual smoke test**

In one terminal:

```bash
poetry run ./start_impsy.py run -c configs/default.toml
```

In a second terminal:

```bash
poetry run ./start_impsy.py webui
```

Open `http://127.0.0.1:4000/realtime` in a browser. With IMPSY running and the model producing output, the green bars should be moving. Move a controller and the cyan bars should follow.

- [ ] **Step 3: Optional — push**

```bash
git push
```
