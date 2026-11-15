#!/usr/bin/env python3
"""Generate placeholder MDRNN models trained on random data.

Trains one model per dimension in DIMENSIONS (default 7, 10, 13, 17, 25) at
MODEL_SIZE "m" (128 units, 2 layers, 5 mixtures) for a single epoch on random
data from `utils.generate_data`. The weights are meaningless; what matters is
that each model has the right shape and goes through IMPSY's normal
train-and-export path.

Use these models as test fixtures for code that loads IMPSY models but doesn't
care what they predict. impsy-auv3 bundles the dim25 model to test loading at
the largest practical size, and impsy-web used the whole set to confirm that
models exported this way contain no Flex ops and run in the browser with
LiteRT.js. Re-running this script after a TensorFlow upgrade is a quick way to
check that still holds.

Edit DIMENSIONS and MODEL_SIZE to make other shapes. Run from the repository
root:

    poetry run python examples/generate_fake_models.py

For each dimension it writes:

    datasets/training-dataset-{dim}d.npz
    models/musicMDRNN-dim{dim}-layers2-units128-mixtures5-scale10.keras
    models/musicMDRNN-dim{dim}-layers2-units128-mixtures5-scale10-ckpt.keras
    models/musicMDRNN-dim{dim}-layers2-units128-mixtures5-scale10.tflite

Existing files with the same names are overwritten.
"""

import numpy as np
from pathlib import Path
from impsy import utils, train

DIMENSIONS = [7, 10, 13, 17, 25]
MODEL_SIZE = "m"
SAMPLES = 5000  # enough to exceed SEQ_LEN=50 after slicing

datasets_dir = Path("datasets")
models_dir = Path("models")
datasets_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

for dim in DIMENSIONS:
    print(f"\n=== Dimension {dim} ===")

    # Generate fake data and save as npz dataset
    data = utils.generate_data(samples=SAMPLES, dimension=dim)
    perfs = np.array([data.astype("float32")], dtype=object)
    dataset_path = datasets_dir / f"training-dataset-{dim}d.npz"
    np.savez_compressed(dataset_path, perfs=perfs)
    print(f"Saved dataset: {dataset_path}")

    # Train model
    train.train_mdrnn(
        dimension=dim,
        dataset_location=str(datasets_dir),
        model_size=MODEL_SIZE,
        early_stopping=False,
        patience=10,
        num_epochs=1,
        batch_size=64,
        save_location=str(models_dir),
        save_model=True,
        save_weights=False,
        save_tflite=True,
    )

print("\nDone.")
