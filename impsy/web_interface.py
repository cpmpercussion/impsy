from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import click
import psutil
import shutil
import platform
import os
import time
import tomllib
from threading import Lock, Thread
from impsy.dataset import generate_dataset
from pathlib import Path
from pythonosc import dispatcher, osc_server
from datetime import datetime

app = Flask(__name__)
app.secret_key = "impsywebui"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(os.path.dirname(CURRENT_DIR))
LOGS_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "datasets"
CONFIGS_DIR = PROJECT_ROOT / "configs"
CONFIG_FILE = "config.toml"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 4000

# Pause toggle state mirrored locally — IMPSY has no return channel, so the
# webui tracks the last commanded state. One click re-syncs if it drifts.
_pause_state = False


def get_version():
    """Get project version from pyproject.toml."""
    try:
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("version", "")
    except Exception:
        return ""


def get_workflow_status():
    """Check which workflow steps have been completed."""
    config_exists = os.path.exists(CONFIG_FILE)

    log_files = list(LOGS_DIR.glob("*.log")) if LOGS_DIR.exists() else []
    dataset_files = list(DATASET_DIR.glob("*.npz")) if DATASET_DIR.exists() else []
    model_files = (
        [f for f in MODEL_DIR.iterdir() if f.suffix in {".keras", ".h5", ".tflite"}]
        if MODEL_DIR.exists()
        else []
    )

    # Try to get dimension from config
    dimension = None
    if config_exists:
        try:
            with open(CONFIG_FILE, "rb") as f:
                config = tomllib.load(f)
            dimension = config.get("model", {}).get("dimension")
        except Exception:
            pass

    return {
        "config": config_exists,
        "logs": len(log_files) > 0,
        "log_count": len(log_files),
        "datasets": len(dataset_files) > 0,
        "dataset_count": len(dataset_files),
        "models": len(model_files) > 0,
        "model_count": len(model_files),
        "dimension": dimension,
    }


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
            if entry[0] == "note_on" and len(entry) >= 2:
                labels.append(f"Note ch{entry[1]}")
            elif entry[0] == "control_change" and len(entry) >= 3:
                labels.append(f"CC{entry[1]}:{entry[2]}")
            else:
                labels.append(f"Ch {i}")
        else:
            labels.append(f"Ch {i}")
    return labels


def get_osc_config():
    """Get OSC configuration if available."""
    try:
        with open(CONFIG_FILE, "rb") as f:
            config = tomllib.load(f)
        osc = config.get("osc", {})
        if osc:
            # server_ip is IMPSY's bind address; for sending commands from this
            # co-located webui, wildcard binds must be redirected to localhost
            # (macOS rejects sends to 0.0.0.0 with "No route to host").
            host = osc.get("server_ip", "127.0.0.1")
            if host in ("0.0.0.0", "::", ""):
                host = "127.0.0.1"
            return {
                "enabled": True,
                "host": host,
                "port": osc.get("server_port", 6000),
            }
    except Exception:
        pass
    return {"enabled": False, "host": "127.0.0.1", "port": 6000}


def format_file_size(size_bytes):
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_file_info(filepath):
    """Get metadata for a file."""
    stat = filepath.stat()
    return {
        "name": filepath.name,
        "size": format_file_size(stat.st_size),
        "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "size_bytes": stat.st_size,
    }


def get_log_file_info(filepath):
    """Get metadata for a log file, including dimension from filename."""
    info = get_file_info(filepath)
    # Parse dimension from filename pattern: *-{dimension}d-mdrnn.log
    name = filepath.stem
    dim = "?"
    parts = name.split("-")
    for part in parts:
        if part.endswith("d") and part[:-1].isdigit():
            dim = part[:-1]
            break
    info["dimension"] = dim
    return info


@app.context_processor
def inject_globals():
    """Inject common template variables."""
    osc = get_osc_config()
    return {
        "workflow": get_workflow_status(),
        "version": get_version(),
        "osc_enabled": osc["enabled"],
        "active_page": "",
    }


def get_hardware_info():
    try:
        cpu_info = platform.machine()
        cpu_cores = psutil.cpu_count(logical=False)
        disk_usage = shutil.disk_usage("/")
        memory = psutil.virtual_memory()
        ram_total = memory.total / (1024**3)
        ram_used = (memory.total - memory.available) / (1024**3)
        disk = disk_usage.free / (1024**3)
        disk_percent = 100 * disk_usage.used / disk_usage.total
        os_info = f"{platform.system()} {platform.release()}"
        return {
            "CPU": cpu_info,
            "CPU Cores": cpu_cores,
            "RAM": f"{ram_used:.1f}/{ram_total:.1f} GB ({memory.percent}%)",
            "Disk Free": f"{disk:.1f} GB ({disk_percent:.0f}% used)",
            "OS": os_info,
        }
    except Exception as e:
        return {"Error": str(e)}


def get_software_info():
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)
    project = pyproject_data.get("project", {})
    urls = project.get("urls", {})
    authors = project.get("authors", [])
    author_names = [a.get("name", "") for a in authors if isinstance(a, dict)]
    return {
        "Project": project.get("name"),
        "Version": project.get("version"),
        "Description": project.get("description"),
        "Authors": author_names if author_names else None,
        "Homepage": urls.get("homepage"),
        "Repository": urls.get("repository"),
    }


def allowed_model_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "keras",
        "h5",
        "tflite",
    }


def allowed_log_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"log"}


def allowed_dataset_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"npz"}


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
_monitor_listener_lock = Lock()


def _ensure_monitor_listener():
    """Lazily build and start the module-level monitor listener.

    Flask defaults to threaded=True, so two simultaneous first requests could
    otherwise race here and double-construct the listener (the second start()
    would fail with port-already-bound). The lock keeps init single-threaded.
    """
    global _monitor_listener
    with _monitor_listener_lock:
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


# === Routes ===


@app.route("/")
def index():
    hardware_info = get_hardware_info()
    software_info = get_software_info()
    return render_template(
        "index.html",
        active_page="dashboard",
        hardware_info=hardware_info,
        software_info=software_info,
    )


@app.route("/logs")
def logs():
    if LOGS_DIR.exists():
        log_files = sorted(
            [get_log_file_info(f) for f in LOGS_DIR.iterdir() if f.suffix == ".log"],
            key=lambda x: x["name"],
            reverse=True,
        )
    else:
        log_files = []
    return render_template("logs.html", active_page="logs", log_files=log_files)


@app.route("/delete_log/<filename>", methods=["POST"])
def delete_log(filename):
    filepath = LOGS_DIR / secure_filename(filename)
    if filepath.exists() and allowed_log_file(filename):
        filepath.unlink()
        flash(f"Deleted {filename}", "success")
    else:
        flash(f"File not found: {filename}", "error")
    return redirect(url_for("logs"))


@app.route("/datasets", methods=["GET", "POST"])
def datasets():
    new_dataset = None
    workflow = get_workflow_status()
    default_dimension = workflow.get("dimension", 9) or 9

    if request.method == "POST":
        dimension = request.form.get("dimension", type=int)
        if dimension:
            try:
                new_dataset_path = generate_dataset(
                    dimension, source=LOGS_DIR, destination=DATASET_DIR
                )
                new_dataset = os.path.basename(new_dataset_path)
                flash(
                    f"Dataset with dimension {dimension} generated successfully!",
                    "success",
                )
            except Exception as e:
                flash(f"Error generating dataset: {str(e)}", "error")
            return redirect(url_for("datasets", new_dataset=new_dataset))

    if DATASET_DIR.exists():
        dataset_files = sorted(
            [get_file_info(f) for f in DATASET_DIR.iterdir() if f.suffix == ".npz"],
            key=lambda x: x["name"],
            reverse=True,
        )
    else:
        dataset_files = []

    new_dataset = request.args.get("new_dataset")
    return render_template(
        "datasets.html",
        active_page="datasets",
        dataset_files=dataset_files,
        new_dataset=new_dataset,
        default_dimension=default_dimension,
    )


@app.route("/delete_dataset/<filename>", methods=["POST"])
def delete_dataset(filename):
    filepath = DATASET_DIR / secure_filename(filename)
    if filepath.exists() and allowed_dataset_file(filename):
        filepath.unlink()
        flash(f"Deleted {filename}", "success")
    else:
        flash(f"File not found: {filename}", "error")
    return redirect(url_for("datasets"))


@app.route("/models", methods=["GET", "POST"])
def models():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_model_file(file.filename):
            filename = secure_filename(file.filename)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            file.save(os.path.join(MODEL_DIR, filename))
            flash(f"Uploaded {filename}", "success")
            return redirect(url_for("models"))

    if MODEL_DIR.exists():
        model_files = sorted(
            [
                get_file_info(f)
                for f in MODEL_DIR.iterdir()
                if allowed_model_file(f.name)
            ],
            key=lambda x: x["name"],
            reverse=True,
        )
    else:
        model_files = []
    return render_template("models.html", active_page="models", model_files=model_files)


@app.route("/delete_model/<filename>", methods=["POST"])
def delete_model(filename):
    filepath = MODEL_DIR / secure_filename(filename)
    if filepath.exists() and allowed_model_file(filename):
        filepath.unlink()
        flash(f"Deleted {filename}", "success")
    else:
        flash(f"File not found: {filename}", "error")
    return redirect(url_for("models"))


@app.route("/download_log/<filename>")
def download_log(filename):
    return send_file(os.path.join(LOGS_DIR, filename), as_attachment=True)


@app.route("/download_model/<filename>")
def download_model(filename):
    return send_file(os.path.join(MODEL_DIR, filename), as_attachment=True)


@app.route("/download_dataset/<filename>")
def download_dataset(filename):
    return send_file(os.path.join(DATASET_DIR, filename), as_attachment=True)


@app.route("/config", methods=["GET", "POST"])
def edit_config():
    config_exists = os.path.exists(CONFIG_FILE)

    if request.method == "POST":
        with open(CONFIG_FILE, "w") as f:
            f.write(request.form["config_content"])
        flash("Configuration saved.", "success")
        return redirect(url_for("edit_config"))

    config_content = ""
    if config_exists:
        with open(CONFIG_FILE, "r") as f:
            config_content = f.read()

    return render_template(
        "edit_config.html",
        active_page="config",
        config_content=config_content,
        config_exists=config_exists,
    )


@app.route("/config/create-default", methods=["POST"])
def create_default_config():
    """Create config.toml from the default template."""
    default_config = CONFIGS_DIR / "default.toml"
    if default_config.exists():
        shutil.copy2(default_config, CONFIG_FILE)
        flash(
            "Created config.toml from default template. Edit it to match your setup.",
            "success",
        )
    else:
        flash("Default config template not found.", "error")
    return redirect(url_for("edit_config"))


@app.route("/config/setup", methods=["GET", "POST"])
def setup_config():
    """Guided configuration setup wizard."""
    if request.method == "POST":
        title = request.form.get("title", "My IMPSY Instrument")
        mode = request.form.get("mode", "callresponse")
        dimension = request.form.get("dimension", type=int, default=9)
        model_size = request.form.get("model_size", "s")
        io_midi = request.form.get("io_midi")
        io_osc = request.form.get("io_osc")
        io_websocket = request.form.get("io_websocket")
        io_serial = request.form.get("io_serial")

        config_lines = [
            f"# IMPSY Configuration",
            f"# Generated by Setup Wizard",
            f"",
            f'title = "{title}"',
            f'owner = ""',
            f'description = ""',
            f"",
            f"log_input = true",
            f"log_predictions = false",
            f"verbose = true",
            f"",
            f"[interaction]",
            f'mode = "{mode}"',
            f"threshold = 0.1",
            f"input_thru = true",
            f"",
            f"[model]",
            f"dimension = {dimension}",
            f'file = ""',
            f'size = "{model_size}"',
            f"sigmatemp = 0.01",
            f"pitemp = 1",
            f"timescale = 1",
        ]

        if io_midi:
            config_lines += [
                f"",
                f"[midi]",
                f'in_device = [""]',
                f'out_device = [""]',
                f"# Configure MIDI input/output mappings for your devices.",
                f"# See configs/ directory for examples.",
            ]

        if io_osc:
            config_lines += [
                f"",
                f"[osc]",
                f'server_ip = "0.0.0.0"',
                f"server_port = 6000",
                f'client_ip = "localhost"',
                f"client_port = 6001",
            ]

        if io_websocket:
            config_lines += [
                f"",
                f"[websocket]",
                f'server_ip = "0.0.0.0"',
                f"server_port = 5001",
            ]

        if io_serial:
            config_lines += [
                f"",
                f"[serial]",
                f'port = "/dev/ttyAMA0"',
                f"baudrate = 115200",
            ]

        config_lines += [
            "",
            "[webui]",
            "monitor_port = 4001  # Localhost port the webui's Realtime page listens on.",
        ]

        config_text = "\n".join(config_lines) + "\n"
        with open(CONFIG_FILE, "w") as f:
            f.write(config_text)

        flash(
            "Configuration created! Edit the details in the config editor.", "success"
        )
        return redirect(url_for("edit_config"))

    return render_template("setup.html", active_page="config")


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


@app.route("/commands", methods=["GET", "POST"])
def commands():
    """Send OSC commands to the running interaction server."""
    global _pause_state
    osc_config = get_osc_config()

    if request.method == "POST":
        command = request.form.get("command", "")
        try:
            from pythonosc import udp_client

            client = udp_client.SimpleUDPClient(osc_config["host"], osc_config["port"])

            if command.startswith("mode:"):
                mode = command.split(":", 1)[1]
                client.send_message("/impsy/mode", mode)
                flash(f"Sent mode change: {mode}", "success")
            elif command == "reset":
                client.send_message("/impsy/reset", 1)
                flash("Sent LSTM reset command.", "success")
            elif command == "pause":
                _pause_state = not _pause_state
                client.send_message("/impsy/pause", 1 if _pause_state else 0)
                flash(
                    f"{'Paused' if _pause_state else 'Resumed'} IMPSY.", "success"
                )
            else:
                flash(f"Unknown command: {command}", "error")
        except ImportError:
            flash("python-osc is required for sending commands.", "error")
        except Exception as e:
            flash(f"Error sending command: {e}", "error")

        return redirect(url_for("commands"))

    return render_template(
        "commands.html",
        active_page="commands",
        osc_host=osc_config["host"],
        osc_port=osc_config["port"],
        pause_active=_pause_state,
    )


def run_web_interface(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=True):
    """Runs the Flask web interface."""
    # Auto-create config.toml from default if it doesn't exist
    if not os.path.exists(CONFIG_FILE):
        default_config = CONFIGS_DIR / "default.toml"
        if default_config.exists():
            shutil.copy2(default_config, CONFIG_FILE)
            click.secho(f"Created config.toml from default template", fg="green")

    click.secho(f"Starting web interface at http://{host}:{port}", fg="blue")
    click.secho(f"Log path: {LOGS_DIR}", fg="blue")
    app.run(host=host, port=port, debug=debug)


@click.command(name="webui")
@click.option("--host", default=DEFAULT_HOST, help="The host to bind to.")
@click.option("--port", default=DEFAULT_PORT, help="The port to bind to.")
@click.option("--debug", is_flag=True, help="Enable debug mode.")
def webui(host, port, debug):
    """Run IMPSY Web UI giving access to files and other commands."""
    run_web_interface(host, port, debug)
