import pytest
import os
import io
from pathlib import Path
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
