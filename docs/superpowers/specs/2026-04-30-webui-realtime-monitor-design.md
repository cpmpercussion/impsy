# Webui Realtime Monitor — Design

**Status:** approved (pending spec self-review)
**Date:** 2026-04-30
**Related issues:** none open; surfaces the underlying need from #46 (verify tutorial workflow) by making it possible to *see* what IMPSY is doing at a glance

## Goal

Add a `/realtime` page to the IMPSY webui that displays incoming and outgoing channel values from the running interaction server as live bar indicators (one row per dimension, in/out side-by-side). Helps users debug their MIDI/OSC mappings, confirm a model is producing sensible output, and visually verify that the server is alive.

The webui and IMPSY are separate processes communicating only over UDP. This feature adds a localhost-only "monitor" UDP channel inside IMPSY that the webui taps; user-configured `[osc]` routing is untouched.

Bundles a fix for a related bug: when IMPSY's `[osc].client_ip` is `"0.0.0.0"` (a wildcard bind, not a valid send destination), macOS rejects sends with `OSError: [Errno 65] No route to host`. Same root cause as the webui-side fix in commit `0bf4367`; same coercion solution.

## Architecture

```
┌──────────────┐    /interface     ┌─────────────────────────────┐
│  controller  │ ───────────────▶ │  InteractionServer          │
│  (Pd / MIDI) │                   │                             │
└──────────────┘                   │  dense_callback ───┐        │
                                   │  send_back_values  │        │
                                   │           │        │        │
                                   │           ▼        ▼        │
                                   │  _broadcast_monitor          │
                                   │  ─── UDP /monitor/{in,out} ──┼──┐
                                   └─────────────────────────────┘  │
                                                                    │
                                              localhost:monitor_port│
                                                                    ▼
                                   ┌─────────────────────────────┐
                                   │  webui (Flask process)      │
                                   │                             │
                                   │  MonitorListener (thread)   │
                                   │   stores latest_in/latest_out│
                                   │                             │
                                   │  GET /realtime  ─── HTML    │
                                   │  GET /realtime/data ─ JSON  │
                                   └─────────────────────────────┘
                                                  ▲
                                                  │ fetch every 100 ms
                                   ┌─────────────────────────────┐
                                   │  browser /realtime page     │
                                   │   sliders update from JSON  │
                                   └─────────────────────────────┘
```

## Components

### 1. IMPSY-side: monitor broadcaster

A new method `InteractionServer._broadcast_monitor(direction: str, values: Iterable[float])` on the existing `InteractionServer` class (no new module; this is a 5-line concern).

- Constructs (or lazily caches) a `pythonosc.udp_client.SimpleUDPClient` targeting `127.0.0.1:<monitor_port>`.
- Sends `/monitor/in` or `/monitor/out` with the float vector.
- Wraps the send in a broad `try/except` — failures (no listener, transient errors) are silently swallowed since the monitor is non-essential and a non-running webui must not break IMPSY.

Hook points (only two — both single call sites in `interaction.py`):

| Hook | Location | What's broadcast |
|---|---|---|
| Inputs | top of `dense_callback` and end of `construct_input_list` | The `dimension - 1` user-input vector (no `dt`) |
| Outputs | top of `send_back_values` | The clipped `dimension - 1` output vector |

The `dt` (first element of `last_user_interaction_data`) is intentionally excluded — sliders show musical parameters in `[0, 1]`, not timing.

### 2. Config: `[webui].monitor_port`

Add a new top-level table `[webui]` with one key, `monitor_port` (int, default `4001`). Both IMPSY and the webui read this — IMPSY uses it as the broadcast destination, webui binds an OSC listener to the same port on localhost.

- Read on the IMPSY side via `self.config.get("webui", {}).get("monitor_port", 4001)`.
- Read on the webui side via the existing `tomllib.load` pattern in `web_interface.py`.
- Both sides default to `4001` if `[webui]` is missing entirely, so existing configs continue to work without edits.
- Updated [`docs/config.md`](../../config.md) with a new `[webui]` section and entries in the OSC addresses table for `/monitor/in`, `/monitor/out`.
- Updated `web_interface.py:setup_config` so the wizard writes a `[webui]` section to new configs.

### 3. Webui-side: `MonitorListener`

A small class in `impsy/web_interface.py`:

```python
class MonitorListener:
    """Lazily-started OSC listener that mirrors IMPSY's /monitor/{in,out} stream."""
    latest_in: list[float] | None
    latest_out: list[float] | None
    in_updated_at: float
    out_updated_at: float

    def __init__(self, port: int): ...
    def start(self) -> None: ...   # idempotent — safe to call from every request
    def stop(self) -> None: ...    # used in app teardown
```

- Owns a `pythonosc.dispatcher.Dispatcher` mapping `/monitor/in` and `/monitor/out` to small callbacks that update the cached arrays + timestamps.
- Owns a `ThreadingOSCUDPServer` and a daemon `Thread`, started by `start()`. `start()` is idempotent.
- Started lazily by the first GET to `/realtime` (so the listener doesn't bind a port at webui boot if no one ever opens the page).
- A single module-level instance (the webui is single-process by design).

### 4. Routes

Two new Flask routes in `web_interface.py`:

| Route | Method | Returns |
|---|---|---|
| `/realtime` | GET | Renders `realtime.html`. Computes channel labels server-side and passes them as a Jinja list. |
| `/realtime/data` | GET | JSON: `{"in": [...], "out": [...], "in_age_ms": N, "out_age_ms": N, "dimension": N}`. `null` arrays if no values seen yet. |

`/realtime` also appears in the navbar (only when `[osc_enabled]`-style guard isn't relevant — monitor works regardless of `[osc]`, so the nav entry is unconditional once the feature ships).

### 5. Channel labels

Computed once at page render, passed into the template:

```
def compute_channel_labels(config: dict) -> list[str]:
    """
    Returns dimension-1 labels for the realtime sliders.
    Pulled from the first MIDI input port's mapping when [midi] exists; else
    falls back to "Ch N".
    """
```

Label rules:
- If `[midi].input` exists and contains at least one device entry, use the *first* device's mapping (sorted by device name for determinism).
  - `["note_on", ch]` → `"Note ch{ch}"`
  - `["control_change", ch, ctrl]` → `"CC{ch}:{ctrl}"` (also handles the 5-arg form `[..., ch, ctrl, min, max]`)
- Else if `[osc]` exists with no MIDI: `["Ch 0", "Ch 1", ..., "Ch {N-1}"]`.
- Else: same fallback (`"Ch N"`).

Multi-port MIDI configs (like `roland-s-1-quneo.toml`) only show the first port's labels. Documented limitation; better than nothing, and the typical case is a single port.

### 6. Browser

`templates/realtime.html` extends `base.html` and contains:

- A `<table>` with one row per channel: `<label>` cell, two cells with horizontal bars (cyan for in, green for out), and a numeric readout cell.
- Bars implemented as plain CSS `width: <pct>%` on a coloured `<div>` inside a 100%-wide track. No external chart library.
- ~40 lines of vanilla JS:
  - `setInterval(poll, 100)` → `fetch('/realtime/data')` → updates each row's bar widths and numeric readouts.
  - "Stale" badge if `*_age_ms > 2000`: row dims to 50% opacity.
  - First-ever-load shows "Waiting for data…" until the first non-null payload arrives.

No interactivity — the bars are display-only. The user described them as "basic sliders" but confirmed display-only intent in brainstorming.

## Data flow

1. User moves a controller (MIDI/OSC).
2. `MIDIServer.handle_port` (or `OSCServer.handle_interface_message`, etc.) calls `InteractionServer.construct_input_list` / `dense_callback`.
3. The callback, before returning, calls `self._broadcast_monitor("in", values)`.
4. UDP packet `/monitor/in <f f ...>` arrives on `127.0.0.1:4001`.
5. Webui's `MonitorListener` thread updates `latest_in` and `in_updated_at`.
6. Browser polling hits `/realtime/data`, gets the latest values, redraws the cyan bars.
7. Symmetrically: when the RNN produces output, `send_back_values` calls `_broadcast_monitor("out", values)` before sending out via the configured I/O servers; the green bars update.

## Bundled bug fix: wildcard `osc.client_ip` coercion

In `OSCServer.__init__` (`impsy/impsio.py`), apply the same coercion the webui uses:

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

The yellow log line is one-shot at startup and tells the user what happened, so they can fix their config if they meant a different destination.

## Error handling

- **No webui running:** IMPSY's `_broadcast_monitor` UDP sends go to a closed port; `SimpleUDPClient.send_message` swallows the resulting `ConnectionRefusedError` on Linux/macOS. We add an outer `try/except Exception` for safety on Windows. Either way, IMPSY keeps running.
- **Webui starts after IMPSY:** First few packets are lost, then the listener picks up. No replay; the page just shows "Waiting for data…" until the next event.
- **Monitor port already in use:** `MonitorListener.start()` raises; the `/realtime` page shows the error message. The rest of the webui is unaffected (lazy start scoped to that one route).
- **Mapping mismatch (config dimension ≠ broadcast vector length):** the realtime page just shows the values as-is; if the dimension changed mid-session the row count won't match but it's a debugging tool, not a precision instrument.

## Testing

| Test | What it covers |
|---|---|
| `test_broadcast_monitor_fires_on_dense_callback` | Mock `_broadcast_monitor`, call `dense_callback`, assert it's called with `("in", values)`. |
| `test_broadcast_monitor_fires_on_send_back_values` | Mock `_broadcast_monitor`, call `send_back_values`, assert `("out", values)`. |
| `test_broadcast_monitor_swallows_errors` | Inject a UDP client whose `send_message` raises; confirm `_broadcast_monitor` doesn't propagate. |
| `test_monitor_listener_roundtrip` | Bind a real `MonitorListener` on a free port, send a fake `/monitor/in` packet via `udp_client.SimpleUDPClient`, assert `latest_in` updates. |
| `test_realtime_data_endpoint` | Flask test-client GET `/realtime/data`, mock the listener's state, assert JSON shape. |
| `test_compute_channel_labels_midi` | Synthetic MIDI config → expected label list. |
| `test_compute_channel_labels_osc_only` | Config with `[osc]` but no `[midi]` → `"Ch 0".."Ch N-1"`. |
| `test_osc_client_ip_wildcard_coerced` | Construct `OSCServer` with `client_ip="0.0.0.0"`, assert the underlying client points at `127.0.0.1`. |

## Out of scope

- **Interactive sliders** (draggable to inject values into IMPSY). Possible follow-up if the user wants the page to also be a control surface; would require `POST /realtime/inject` and an OSC sender from webui → IMPSY's input port.
- **History/trail** (last N seconds of values as a sparkline). Adds complexity; current values are sufficient for the stated debugging goal.
- **Per-channel labels driven by config comments** (e.g. the `# osc square level` style notes in `roland-s-1.toml`). TOML comments aren't accessible via `tomllib`; would need either a structured `name` field per mapping entry or a separate label list. Defer until requested.
- **Multi-port MIDI labels.** Only the first port's mapping is used. Multi-port configs see partial labels with the rest as `"Ch N"` — acceptable for first version.
- **Authentication** on the realtime page. The webui has no auth anywhere; this matches.

## Documentation updates

- `docs/config.md` — add `[webui]` section and document `monitor_port`. Add `/monitor/in` and `/monitor/out` to the OSC addresses table (mark as "internal — emitted by IMPSY when `[webui].monitor_port` is configured; consumed by the webui's realtime page").
- The realtime page itself includes one paragraph of help text describing what the bars represent.

## Open questions

None as of writing — all three brainstorming clarifying questions resolved before this spec was written.
