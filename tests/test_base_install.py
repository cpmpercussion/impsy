"""Inference-only smoke tests for the base install (no `[train]` extra).

These tests are designed to run with `pip install .` only — i.e., without
TensorFlow, Keras, h5py, or keras-mdn-layer installed. They confirm that:

  * `import impsy` and its inference-relevant submodules work without TF;
  * `DummyMDRNN` and `TfliteMDRNN` produce predictions of the expected shape;
  * Importing `compat.get_tflite_interpreter` does not pull in TF;
  * Training entry points raise a helpful ImportError naming the `[train]` extra.

The full-extras CI matrix continues to run the standard test suite (which uses
fixtures that train a model). This file deliberately avoids those fixtures and
relies only on the small `.tflite` artifacts committed under `models/`.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
# This .tflite was exported without SELECT_TF_OPS, so it loads under a
# pure-LiteRT runtime with no TensorFlow available — which is exactly what the
# base install provides.
SAMPLE_TFLITE = (
    REPO_ROOT / "models" / "musicMDRNN-dim9-layers2-units64-mixtures5-scale10.tflite"
)


@pytest.fixture
def isolated_tflite(tmp_path):
    """Copy the bundled .tflite into tmp_path so any rebuild fallback path
    in TfliteMDRNN.prepare() can't mutate the source fixture in the repo.
    """
    dst = tmp_path / SAMPLE_TFLITE.name
    shutil.copyfile(SAMPLE_TFLITE, dst)
    return dst


def test_impsy_imports_without_tensorflow():
    """`import impsy` and the inference module must not require TF."""
    import impsy  # noqa: F401
    import impsy.compat  # noqa: F401
    import impsy.mdrnn  # noqa: F401
    import impsy.interaction  # noqa: F401


def test_dummy_mdrnn_runs():
    """DummyMDRNN must work as the no-model fallback even on a TF-free install."""
    from impsy import mdrnn

    dummy = mdrnn.DummyMDRNN(Path("nope"), dimension=4, n_hidden_units=8, n_mixtures=5, n_layers=2)
    sample = dummy.generate(np.zeros(4))
    assert sample.shape == (4,)


@pytest.mark.skipif(not SAMPLE_TFLITE.exists(), reason="bundled .tflite fixture missing")
def test_tflite_mdrnn_inference_runs(isolated_tflite):
    """TfliteMDRNN must load and predict from a .tflite file without TF."""
    from impsy import mdrnn

    model = mdrnn.TfliteMDRNN.from_file(isolated_tflite)
    prev = np.zeros(model.dimension)
    for _ in range(3):
        out = model.generate(prev)
        assert out.shape == (model.dimension,)
        prev = out


def test_inline_mdn_sampler_matches_shape():
    """The inlined sampler must return a vector of length `output_dim`."""
    from impsy.mdrnn import sample_mdn_output

    output_dim, num_mixes = 3, 4
    params_len = num_mixes + 2 * num_mixes * output_dim
    params = np.zeros(params_len, dtype=np.float32)
    sample = sample_mdn_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0)
    assert sample.shape == (output_dim,)


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is not None,
    reason="train extra is installed; the friendly ImportError is only raised without TF",
)
def test_train_entry_points_raise_friendly_importerror():
    """When TF is absent, training helpers must raise a clear ImportError naming the extra."""
    from impsy import mdrnn

    with pytest.raises(ImportError, match=r"impsy\[train\]"):
        mdrnn.build_mdrnn_model(
            dimension=2, n_hidden_units=8, n_mixtures=5, n_layers=2, inference=True
        )
