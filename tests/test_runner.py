"""
Tests for run_test() in evaluation/runner.py.

Covers:
- Top-level output files are created (test_metrics.json, test_metrics.csv, test_log.txt)
- Per-patient directories with prediction.json / xai.json / xai_report.txt
- prediction.json has all required keys with valid values
- test_metrics.json contains the expected metric keys
- With-complications variant writes complication_probs to prediction.json

Heavy data-loading dependencies (load_splits_and_config, build_data_list, CacheDataset,
DataLoader, get_cached_transforms, get_val_runtime_transforms) are mocked so the test
runs CPU-only without deepstore access.  The model and checkpoint are real.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tests.conftest import AAA_DIM, CLINICAL_DIM, COMPLICATIONS, N_COMP, SPATIAL


# ── force CPU even on GPU-allocated SLURM nodes ───────────────────────────────
# runner.py's module-level DEVICE resolves at import time.  Patching it keeps
# the model that run_test() builds on CPU, matching the synthetic batch tensors.

@pytest.fixture(autouse=True)
def _force_cpu():
    with patch("src.evaluation.runner.DEVICE", torch.device("cpu")):
        yield


# ── helpers ───────────────────────────────────────────────────────────────────

FEATURE_NAMES = [f"feat_{i}" for i in range(CLINICAL_DIM)]


def _make_checkpoint(tmp_path, model, feature_names, complications):
    """Save a real checkpoint in the exact format that trainer.py produces."""
    ckpt = {
        "state_dict":    model.state_dict(),
        "threshold":     0.5,
        "feature_names": feature_names,
        "clin_mean":     [0.0] * len(feature_names),
        "clin_std":      [1.0] * len(feature_names),
        "complications": complications,
        "config":        {},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(ckpt, path)
    return str(path)


def _make_test_batch(pid: str = "P001", n_comp: int = 0) -> dict:
    """Synthetic single-patient batch (batch_size=1) as DataLoader would yield."""
    batch = {
        "pid":                  [pid],
        "cta":                  torch.rand(1, 4, SPATIAL, SPATIAL, SPATIAL),
        "clin":                 torch.rand(1, CLINICAL_DIM),
        "aaa_metrics":          torch.rand(1, AAA_DIM),
        "reintervention_label": torch.FloatTensor([1.0]),
    }
    if n_comp:
        batch["complication_labels"] = torch.zeros(1, n_comp)
    return batch


def _base_cfg(base_dir: str) -> dict:
    return {
        "name": "test_run",
        "model": {"aaa_dim": AAA_DIM, "complications": []},
        "training": {
            "image_size":    [SPATIAL, SPATIAL, SPATIAL],
            "voxel_spacing": [1.5, 1.5, 1.5],
        },
        "paths": {"base_dir": base_dir},
    }


def _patch_runner(batches: list, feature_names: list, split_ids: list):
    """Return a context manager stacking all six data-loading patches for run_test()."""
    from contextlib import ExitStack
    from unittest.mock import patch as _patch

    stack = ExitStack()
    mocks = {}
    for name in (
        "load_splits_and_config",
        "build_data_list",
        "get_cached_transforms",
        "get_val_runtime_transforms",
        "CacheDataset",
        "DataLoader",
    ):
        mocks[name] = stack.enter_context(_patch(f"src.evaluation.runner.{name}"))

    mocks["load_splits_and_config"].return_value = (
        {"train": [], "val": [], "test": split_ids}, {}, feature_names
    )
    mocks["build_data_list"].return_value = [{}] * len(split_ids)
    mocks["get_cached_transforms"].return_value = lambda x: x
    mocks["get_val_runtime_transforms"].return_value = lambda x: x
    mocks["CacheDataset"].return_value = MagicMock()
    mocks["DataLoader"].return_value = batches
    return stack


# ── run_test() — no-complications ─────────────────────────────────────────────


def test_run_test_creates_top_level_files(tmp_path, fresh_model_no_comp):
    from src.evaluation.runner import run_test

    ckpt_path = _make_checkpoint(tmp_path, fresh_model_no_comp, FEATURE_NAMES, [])
    out_dir = tmp_path / "out"
    batches = [_make_test_batch("P001"), _make_test_batch("P002")]
    batches[1]["reintervention_label"] = torch.FloatTensor([0.0])

    with _patch_runner(batches, FEATURE_NAMES, ["P001", "P002"]):
        run_test(_base_cfg(str(tmp_path)), ckpt_path, str(out_dir))

    assert (out_dir / "test_metrics.json").exists()
    assert (out_dir / "test_metrics.csv").exists()
    assert (out_dir / "test_log.txt").exists()


def test_run_test_creates_per_patient_files(tmp_path, fresh_model_no_comp):
    from src.evaluation.runner import run_test

    ckpt_path = _make_checkpoint(tmp_path, fresh_model_no_comp, FEATURE_NAMES, [])
    out_dir = tmp_path / "out"
    batches = [_make_test_batch("P001"), _make_test_batch("P002")]
    batches[1]["reintervention_label"] = torch.FloatTensor([0.0])

    with _patch_runner(batches, FEATURE_NAMES, ["P001", "P002"]):
        run_test(_base_cfg(str(tmp_path)), ckpt_path, str(out_dir))

    for pid in ("P001", "P002"):
        assert (out_dir / pid / "prediction.json").exists()
        assert (out_dir / pid / "xai.json").exists()
        assert (out_dir / pid / "xai_report.txt").exists()


def test_run_test_prediction_json_format(tmp_path, fresh_model_no_comp):
    from src.evaluation.runner import run_test

    ckpt_path = _make_checkpoint(tmp_path, fresh_model_no_comp, FEATURE_NAMES, [])
    out_dir = tmp_path / "out"
    batches = [_make_test_batch("P001")]

    with _patch_runner(batches, FEATURE_NAMES, ["P001"]):
        run_test(_base_cfg(str(tmp_path)), ckpt_path, str(out_dir))

    pred = json.loads((out_dir / "P001" / "prediction.json").read_text())
    required = {
        "pid", "reintervention_prob", "reintervention_pred",
        "reintervention_gt", "label_type", "threshold", "reint_logvar",
    }
    assert required <= pred.keys()
    assert pred["pid"] == "P001"
    assert 0.0 <= pred["reintervention_prob"] <= 1.0
    assert pred["reintervention_pred"] in (0, 1)
    assert pred["label_type"] in ("TP", "FP", "TN", "FN")
    assert pred["threshold"] == pytest.approx(0.5)


def test_run_test_metrics_json_keys(tmp_path, fresh_model_no_comp):
    from src.evaluation.runner import run_test

    ckpt_path = _make_checkpoint(tmp_path, fresh_model_no_comp, FEATURE_NAMES, [])
    out_dir = tmp_path / "out"
    batches = [_make_test_batch("P001"), _make_test_batch("P002")]
    batches[1]["reintervention_label"] = torch.FloatTensor([0.0])

    with _patch_runner(batches, FEATURE_NAMES, ["P001", "P002"]):
        run_test(_base_cfg(str(tmp_path)), ckpt_path, str(out_dir))

    metrics = json.loads((out_dir / "test_metrics.json").read_text())
    expected_keys = {
        "n_samples", "auc", "f1", "precision", "recall",
        "accuracy", "threshold_used",
    }
    assert expected_keys <= metrics.keys()
    assert metrics["n_samples"] == 2


# ── run_test() — with-complications ───────────────────────────────────────────


def test_run_test_with_comp_prediction_has_comp_probs(tmp_path, fresh_model_with_comp):
    from src.evaluation.runner import run_test

    ckpt_path = _make_checkpoint(tmp_path, fresh_model_with_comp, FEATURE_NAMES, COMPLICATIONS)
    cfg = _base_cfg(str(tmp_path))
    cfg["model"]["complications"] = COMPLICATIONS
    out_dir = tmp_path / "out"
    batches = [_make_test_batch("P001", n_comp=N_COMP)]

    with _patch_runner(batches, FEATURE_NAMES, ["P001"]):
        run_test(cfg, ckpt_path, str(out_dir))

    pred = json.loads((out_dir / "P001" / "prediction.json").read_text())
    assert "complication_probs" in pred
    for name in COMPLICATIONS:
        assert name in pred["complication_probs"]
        assert 0.0 <= pred["complication_probs"][name] <= 1.0


def test_run_test_with_comp_patient_files_exist(tmp_path, fresh_model_with_comp):
    from src.evaluation.runner import run_test

    ckpt_path = _make_checkpoint(tmp_path, fresh_model_with_comp, FEATURE_NAMES, COMPLICATIONS)
    cfg = _base_cfg(str(tmp_path))
    cfg["model"]["complications"] = COMPLICATIONS
    out_dir = tmp_path / "out"
    batches = [_make_test_batch("P001", n_comp=N_COMP)]

    with _patch_runner(batches, FEATURE_NAMES, ["P001"]):
        run_test(cfg, ckpt_path, str(out_dir))

    assert (out_dir / "P001" / "prediction.json").exists()
    assert (out_dir / "P001" / "xai.json").exists()
    assert (out_dir / "P001" / "xai_report.txt").exists()
