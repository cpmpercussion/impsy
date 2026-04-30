import pytest
import os
import io
import time
from pathlib import Path
from pythonosc import udp_client
from impsy.web_interface import (
    app,
    allowed_model_file,
    allowed_log_file,
    allowed_dataset_file,
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"IMPSY" in response.data
    assert b"Dashboard" in response.data


def test_logs_route(client):
    response = client.get("/logs")
    assert response.status_code == 200


def test_config_route(client):
    response = client.get("/config")
    assert response.status_code == 200


def test_datasets_get(client):
    """Test GET /datasets returns 200."""
    response = client.get("/datasets")
    assert response.status_code == 200


def test_models_get(client):
    """Test GET /models returns 200."""
    response = client.get("/models")
    assert response.status_code == 200


def test_setup_get(client):
    """Test GET /config/setup returns 200."""
    response = client.get("/config/setup")
    assert response.status_code == 200


# File extension validators


def test_allowed_model_file():
    assert allowed_model_file("model.keras") is True
    assert allowed_model_file("model.h5") is True
    assert allowed_model_file("model.tflite") is True
    assert allowed_model_file("model.txt") is False
    assert allowed_model_file("noext") is False
    assert allowed_model_file("model.KERAS") is True  # case insensitive


def test_allowed_log_file():
    assert allowed_log_file("data.log") is True
    assert allowed_log_file("data.txt") is False
    assert allowed_log_file("noext") is False


def test_allowed_dataset_file():
    assert allowed_dataset_file("data.npz") is True
    assert allowed_dataset_file("data.csv") is False
    assert allowed_dataset_file("noext") is False


# POST routes


def test_models_upload_valid(client, tmp_path):
    """Test uploading a valid .keras model file."""
    from impsy.web_interface import MODEL_DIR

    # Create the model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    data = {"file": (io.BytesIO(b"fake model data"), "test_upload.keras")}
    response = client.post("/models", data=data, content_type="multipart/form-data")
    assert response.status_code == 302  # redirect after upload
    # Clean up
    test_file = os.path.join(MODEL_DIR, "test_upload.keras")
    if os.path.exists(test_file):
        os.remove(test_file)


def test_models_upload_invalid_extension(client):
    """Test uploading a file with invalid extension is not saved."""
    from impsy.web_interface import MODEL_DIR

    data = {"file": (io.BytesIO(b"bad data"), "test_upload.exe")}
    response = client.post("/models", data=data, content_type="multipart/form-data")
    assert response.status_code == 200  # renders the models page without redirect
    # Verify the file was NOT saved
    assert not os.path.exists(os.path.join(MODEL_DIR, "test_upload.exe"))


def test_models_upload_no_file(client):
    """Test POST without file redirects back."""
    response = client.post("/models", data={}, content_type="multipart/form-data")
    assert response.status_code == 302


def test_models_upload_empty_filename(client):
    """Test POST with empty filename redirects back."""
    data = {"file": (io.BytesIO(b""), "")}
    response = client.post("/models", data=data, content_type="multipart/form-data")
    assert response.status_code == 302


def test_config_post(client, tmp_path):
    """Test POST /config saves content and redirects."""
    from impsy.web_interface import CONFIG_FILE

    config_backup = None
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config_backup = f.read()

    try:
        response = client.post("/config", data={"config_content": 'title = "test"\n'})
        assert response.status_code == 302  # redirect after save
    finally:
        # Restore the original config
        if config_backup is not None:
            with open(CONFIG_FILE, "w") as f:
                f.write(config_backup)


def test_datasets_post(client):
    """Test POST /datasets with dimension triggers dataset generation."""
    response = client.post("/datasets", data={"dimension": "2"}, follow_redirects=True)
    assert response.status_code == 200


def test_setup_post(client):
    """Test POST /config/setup creates a config file and redirects."""
    from impsy.web_interface import CONFIG_FILE

    config_backup = None
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config_backup = f.read()

    try:
        response = client.post(
            "/config/setup",
            data={
                "title": "Test Instrument",
                "mode": "callresponse",
                "dimension": "5",
                "model_size": "s",
                "io_osc": "1",
            },
        )
        assert response.status_code == 302
        assert os.path.exists(CONFIG_FILE)
        with open(CONFIG_FILE, "r") as f:
            content = f.read()
        assert "Test Instrument" in content
        assert "dimension = 5" in content
        assert "[osc]" in content
    finally:
        if config_backup is not None:
            with open(CONFIG_FILE, "w") as f:
                f.write(config_backup)


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
        assert listener.latest_in == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)
        assert listener.in_updated_at > 0

        client.send_message("/monitor/out", [0.4, 0.5])
        deadline = time.time() + 1.0
        while time.time() < deadline and listener.latest_out is None:
            time.sleep(0.02)
        assert listener.latest_out == pytest.approx([0.4, 0.5], rel=1e-5)
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


def test_compute_channel_labels_handles_malformed_entry():
    """Truncated/malformed mapping entries fall back to numeric labels rather
    than crashing the realtime page with an IndexError."""
    from impsy.web_interface import compute_channel_labels

    cfg = {
        "model": {"dimension": 4},
        "midi": {
            "input": {
                "Foo": [
                    ["note_on"],            # missing channel
                    ["control_change", 1],  # missing controller number
                    ["unknown_kind", 1, 2],  # unrecognised type
                ]
            }
        },
    }
    labels = compute_channel_labels(cfg)
    assert labels == ["Ch 0", "Ch 1", "Ch 2"]


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
