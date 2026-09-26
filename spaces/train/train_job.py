"""Makes a dataset from a folder of logs and trains a model, in its own process.

Run as: python train_job.py WORKDIR DIMENSION SIZE EPOCHS PATIENCE

Reads WORKDIR/logs/*.log, writes WORKDIR/datasets and WORKDIR/models, and
writes WORKDIR/result.json with the .tflite path and loss history.
"""

import json
import sys
from pathlib import Path

import numpy as np
from impsy.dataset import generate_dataset
from impsy.train import SEQ_LEN, train_mdrnn

workdir = Path(sys.argv[1])
dimension = int(sys.argv[2])
size = sys.argv[3]
epochs = int(sys.argv[4])
patience = int(sys.argv[5])

(workdir / "datasets").mkdir(exist_ok=True)
(workdir / "models").mkdir(exist_ok=True)

print(f"Making a dataset from the {dimension}d logs.", flush=True)
dataset_file = generate_dataset(
    dimension, source=str(workdir / "logs"), destination=str(workdir / "datasets")
)
if dataset_file is None:
    sys.exit(
        f"No data found in the {dimension}d logs. Logs need rows recorded from "
        "your interface (with 'interface' in the second column)."
    )
with np.load(dataset_file, allow_pickle=True) as loaded:
    long_enough = [p for p in loaded["perfs"] if len(p) > SEQ_LEN + 1]
print(f"{len(long_enough)} log(s) long enough to train on.", flush=True)
if not long_enough:
    sys.exit(
        f"None of your logs have more than {SEQ_LEN + 1} events, "
        "so there's nothing to train on. Record some longer logs."
    )

output = train_mdrnn(
    dimension,
    dataset_file,
    size,
    early_stopping=True,
    patience=patience,
    num_epochs=epochs,
    batch_size=64,
    save_location=str(workdir / "models"),
)

history = output["history"].history
(workdir / "result.json").write_text(
    json.dumps(
        {
            "tflite_file": str(output["tflite_file"]),
            "loss": [float(v) for v in history.get("loss", [])],
            "val_loss": [float(v) for v in history.get("val_loss", [])],
        }
    )
)
