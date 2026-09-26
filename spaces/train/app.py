"""IMPSY model training on Hugging Face Spaces.

Upload .log files, train an IMPSY model on them, and download a .tflite file.
Each run works in its own temporary folder, which is deleted when the run
finishes, so uploaded logs aren't kept.
"""

import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import gradio as gr
import pandas as pd

MODEL_SIZES = ["xxs", "xs", "s", "m"]  # l and xl are too slow on a free CPU.
MAX_EPOCHS = 200
TIMEOUT_SECONDS = 30 * 60
LOG_NAME = re.compile(r"-(\d+)d-mdrnn\.log$")
TRAIN_JOB = Path(__file__).parent / "train_job.py"
DOWNLOADS = Path(tempfile.gettempdir()) / "impsy-models"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
EPOCH_SUMMARY = re.compile(r"loss: ([\d.]+) - val_loss: ([\d.]+)")
# Lines from impsy's dataset and training steps worth showing.
SHOW_PREFIXES = (
    "Making a dataset",
    "total number of interactions",
    "total time represented",
    "log(s) long enough",
    "Number of training examples",
    "No data found",
    "None of your logs",
)


def collect_logs(files, logs_dir: Path) -> list[str]:
    """Copy uploaded .log files (or .log files inside .zip files) into logs_dir."""
    notes = []
    for f in files or []:
        path = Path(f)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():
                    if member.endswith(".log") and not member.startswith("__MACOSX"):
                        (logs_dir / Path(member).name).write_bytes(z.read(member))
        elif path.suffix == ".log":
            shutil.copy(path, logs_dir / path.name)
        else:
            notes.append(f"Skipping {path.name}: not a .log or .zip file.")
    return notes


def write_example_logs(logs_dir: Path, dimension: int = 4, count: int = 3):
    """Write a few logs of wandering sine waves, for trying the Space out."""
    for n in range(count):
        start = datetime(2026, 1, 1, 12, n * 10)
        t = start
        phases = [random.uniform(0, 2 * math.pi) for _ in range(dimension - 1)]
        lines = []
        for i in range(1000):
            t += timedelta(seconds=random.uniform(0.05, 0.3))
            values = [
                0.5 + 0.45 * math.sin(i / (8 + 4 * j) + phases[j]) for j in range(dimension - 1)
            ]
            values = [min(1, max(0, v + random.gauss(0, 0.02))) for v in values]
            lines.append(f"{t.isoformat()},interface," + ",".join(f"{v:.4f}" for v in values))
        name = start.strftime("%Y-%m-%dT%H-%M-%S") + f"-{dimension}d-mdrnn.log"
        (logs_dir / name).write_text("\n".join(lines) + "\n")


def train(files, use_example, size, epochs, patience, dimension):
    """Run one training job, yielding (log text, loss plot, model file) as it goes."""
    text = ""

    def say(line):
        nonlocal text
        text += line + "\n"
        return text

    with tempfile.TemporaryDirectory(prefix="impsy-") as tmp:
        workdir = Path(tmp)
        logs_dir = workdir / "logs"
        logs_dir.mkdir()

        if use_example:
            write_example_logs(logs_dir)
            say("Using example data: three 4d logs of wandering sine waves.")
        for note in collect_logs(files, logs_dir):
            say(note)

        dimensions = Counter()
        for log in sorted(logs_dir.glob("*.log")):
            match = LOG_NAME.search(log.name)
            if match:
                dimensions[int(match.group(1))] += 1
            else:
                say(f"Ignoring {log.name}: its name should end in -{{dimension}}d-mdrnn.log")
        if not dimensions:
            yield say("No usable log files. Upload some .log files, or tick 'Use example data'."), None, None
            return
        for dim, count in sorted(dimensions.items()):
            say(f"{dim}d: {count} log file(s)")

        dimension = int(dimension or 0) or dimensions.most_common(1)[0][0]
        if dimension not in dimensions:
            yield say(f"There are no {dimension}d logs. Set dimension to 0 to choose automatically."), None, None
            return
        say(f"Training a {size} model on {dimension}d data (at most {int(epochs)} epochs).\n")
        yield text, None, None

        proc = subprocess.Popen(
            [sys.executable, "-u", str(TRAIN_JOB), str(workdir), str(dimension), size, str(int(epochs)), str(int(patience))],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started = datetime.now()
        raw = []  # everything the job printed, shown if it fails
        epoch = 0
        rows = []
        for line in proc.stdout:
            line = ANSI.sub("", line).rstrip()
            raw.append(line)
            if line.startswith("Epoch ") and "/" in line.split()[1]:
                epoch = int(line.split()[1].split("/")[0])
            elif match := EPOCH_SUMMARY.search(line):
                loss, val_loss = float(match.group(1)), float(match.group(2))
                rows += [
                    {"epoch": epoch, "loss": loss, "set": "training"},
                    {"epoch": epoch, "loss": val_loss, "set": "validation"},
                ]
                yield say(f"Epoch {epoch}: loss {loss:.3f}, validation loss {val_loss:.3f}"), pd.DataFrame(rows), None
            elif "early stopping" in line:
                yield say(f"Stopped early: validation loss hasn't improved for {int(patience)} epochs."), pd.DataFrame(rows), None
            elif line.startswith("Do the conversion"):
                yield say("Converting to .tflite."), pd.DataFrame(rows), None
            elif line.startswith(SHOW_PREFIXES):
                yield say(line), (pd.DataFrame(rows) if rows else None), None
            if (datetime.now() - started).total_seconds() > TIMEOUT_SECONDS:
                proc.kill()
                yield say("\nStopped: training took longer than 30 minutes. Try fewer epochs or a smaller model."), None, None
                return
        if proc.wait() != 0 or not (workdir / "result.json").exists():
            say("\nTraining failed. The last messages were:\n")
            yield say("\n".join(raw[-30:])), None, None
            return

        result = json.loads((workdir / "result.json").read_text())
        # Copy the model out of the temporary folder so Gradio can serve it.
        DOWNLOADS.mkdir(exist_ok=True)
        out_dir = Path(tempfile.mkdtemp(dir=DOWNLOADS))
        model = Path(shutil.copy(result["tflite_file"], out_dir))
        say(f"\nDone! Download {model.name} below.")
        yield text, pd.DataFrame(rows), str(model)


with gr.Blocks(title="IMPSY model training") as demo:
    gr.Markdown(
        """
# Train an IMPSY model

Upload `.log` files from [IMPSY](https://charlesmartin.au/impsy), IMPSYpi, the IMPSY AUv3 app or IMPSY Web,
and this trains a model you can perform with. Their names end in `-{dimension}d-mdrnn.log`.
You can also upload a `.zip` of logs. Your logs are deleted as soon as training finishes.

Training a small model on about 10 minutes of data takes a few minutes. Only one model trains at a time,
so you might have to wait in a queue.
"""
    )
    with gr.Row():
        with gr.Column():
            files = gr.File(label="Log files", file_count="multiple", file_types=[".log", ".zip"])
            use_example = gr.Checkbox(label="Use example data (if you don't have logs yet)")
            size = gr.Dropdown(MODEL_SIZES, value="s", label="Model size", info="xs or s is a good start.")
            with gr.Accordion("More settings", open=False):
                epochs = gr.Slider(1, MAX_EPOCHS, value=100, step=1, label="Max epochs")
                patience = gr.Slider(1, 30, value=10, step=1, label="Patience",
                                     info="Stop when validation loss hasn't improved for this many epochs.")
                dimension = gr.Number(0, precision=0, label="Dimension",
                                      info="0 uses the dimension with the most log files.")
            button = gr.Button("Train", variant="primary")
        with gr.Column():
            output_log = gr.Textbox(label="Progress", lines=18, max_lines=18, autoscroll=True)
            plot = gr.LinePlot(x="epoch", y="loss", color="set", label="Loss (lower is better)")
            model_file = gr.File(label="Your model (.tflite)")
    gr.Markdown(
        "Put the `.tflite` file in your IMPSY `models` folder (or upload it on the web UI's Models page) "
        "and set `file`, `dimension` and `size` under `[model]` in `config.toml`. "
        "In IMPSY AUv3 and IMPSY Web, load the `.tflite` file in the app."
    )
    button.click(train, [files, use_example, size, epochs, patience, dimension], [output_log, plot, model_file])

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch()
