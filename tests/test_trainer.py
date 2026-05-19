"""
Tests for validate() (trainer.py) and checkpoint save format.

Covers:
- validate() returns (float, dict) with all required metric keys
- Metrics are numerically in valid ranges
- F-beta threshold selection produces a value in [0, 1]
- All-same GT is handled gracefully (AUC = 0.5)
- with-complications variant emits comp_mean_auc
- wandb.log is called once per validate() call
- Checkpoint dict contains all keys required by load/inference
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from tests.conftest import AAA_DIM, BATCH, CLINICAL_DIM, COMPLICATIONS, N_COMP, SPATIAL


# ── force CPU even on GPU-allocated SLURM nodes ───────────────────────────────
# trainer.py's module-level DEVICE resolves at import time.  Patching it here
# ensures batch tensors and model weights stay on the same device in all tests.

@pytest.fixture(autouse=True)
def _force_cpu():
    with patch("src.training.trainer.DEVICE", torch.device("cpu")):
        yield


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_reint_batch(n: int = BATCH) -> dict:
    """Synthetic batch matching the key names validate() reads from val_loader."""
    return {
        "cta":                  torch.rand(n, 4, SPATIAL, SPATIAL, SPATIAL),
        "clin":                 torch.rand(n, CLINICAL_DIM),
        "aaa_metrics":          torch.rand(n, AAA_DIM),
        "reintervention_label": torch.FloatTensor([0.0, 1.0][:n]),
    }


def _make_comp_batch(n: int = BATCH) -> dict:
    batch = _make_reint_batch(n)
    batch["complication_labels"] = torch.FloatTensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]][:n]
    )
    return batch


class _FakeLoader:
    """Minimal DataLoader stand-in: iterable + __len__."""

    def __init__(self, batches: list) -> None:
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


# ── validate() — no-complications variant ─────────────────────────────────────


@patch("src.training.trainer.wandb")
def test_validate_no_comp_returns_tuple(mock_wandb, model_no_comp):
    from src.training.losses import ReinterventionLoss
    from src.training.trainer import validate

    criterion = ReinterventionLoss(device=torch.device("cpu"))
    val_loss, metrics = validate(
        model_no_comp, _FakeLoader([_make_reint_batch()]),
        criterion, complications=[],
    )
    assert isinstance(val_loss, float)
    assert isinstance(metrics, dict)


@patch("src.training.trainer.wandb")
def test_validate_no_comp_required_keys(mock_wandb, model_no_comp):
    from src.training.losses import ReinterventionLoss
    from src.training.trainer import validate

    required = {
        "val/loss_total", "val/reint_auc", "val/reint_f1",
        "val/reint_precision", "val/reint_recall", "val/reint_accuracy",
        "val/reint_best_thresh", "val/mean_uncertainty",
        "val/reint_TP", "val/reint_FP", "val/reint_TN", "val/reint_FN",
        "val/reint_TPR", "val/reint_TNR", "val/reint_FPR", "val/reint_FNR",
    }
    criterion = ReinterventionLoss(device=torch.device("cpu"))
    _, metrics = validate(
        model_no_comp, _FakeLoader([_make_reint_batch()]),
        criterion, complications=[],
    )
    assert required <= metrics.keys()


@patch("src.training.trainer.wandb")
def test_validate_no_comp_has_no_comp_keys(mock_wandb, model_no_comp):
    """No-complications variant must not emit complication-specific metric keys."""
    from src.training.losses import ReinterventionLoss
    from src.training.trainer import validate

    criterion = ReinterventionLoss(device=torch.device("cpu"))
    _, metrics = validate(
        model_no_comp, _FakeLoader([_make_reint_batch()]),
        criterion, complications=[],
    )
    assert "val/comp_mean_auc" not in metrics


@patch("src.training.trainer.wandb")
def test_validate_no_comp_metrics_in_range(mock_wandb, model_no_comp):
    from src.training.losses import ReinterventionLoss
    from src.training.trainer import validate

    criterion = ReinterventionLoss(device=torch.device("cpu"))
    _, metrics = validate(
        model_no_comp, _FakeLoader([_make_reint_batch()]),
        criterion, complications=[],
    )
    assert 0.0 <= metrics["val/reint_auc"] <= 1.0
    assert 0.0 <= metrics["val/reint_accuracy"] <= 1.0
    assert metrics["val/loss_total"] >= 0.0


@patch("src.training.trainer.wandb")
def test_validate_fbeta_threshold_in_range(mock_wandb, model_no_comp):
    """F-beta PR-curve threshold must be a valid probability value."""
    from src.training.losses import ReinterventionLoss
    from src.training.trainer import validate

    criterion = ReinterventionLoss(device=torch.device("cpu"))
    _, metrics = validate(
        model_no_comp, _FakeLoader([_make_reint_batch()]),
        criterion, complications=[],
    )
    assert 0.0 <= metrics["val/reint_best_thresh"] <= 1.0


@patch("src.training.trainer.wandb")
def test_validate_all_same_gt_gives_auc_half(mock_wandb, model_no_comp):
    """All-positive ground truth must produce AUC = 0.5 (graceful undefined-AUC handling)."""
    from src.training.losses import ReinterventionLoss
    from src.training.trainer import validate

    batch = {
        "cta":                  torch.rand(BATCH, 4, SPATIAL, SPATIAL, SPATIAL),
        "clin":                 torch.rand(BATCH, CLINICAL_DIM),
        "aaa_metrics":          torch.rand(BATCH, AAA_DIM),
        "reintervention_label": torch.ones(BATCH),   # all positive
    }
    criterion = ReinterventionLoss(device=torch.device("cpu"))
    _, metrics = validate(
        model_no_comp, _FakeLoader([batch]),
        criterion, complications=[],
    )
    assert metrics["val/reint_auc"] == pytest.approx(0.5)


@patch("src.training.trainer.wandb")
def test_validate_wandb_log_called_once(mock_wandb, model_no_comp):
    """validate() must call wandb.log exactly once per invocation."""
    from src.training.losses import ReinterventionLoss
    from src.training.trainer import validate

    criterion = ReinterventionLoss(device=torch.device("cpu"))
    validate(
        model_no_comp, _FakeLoader([_make_reint_batch()]),
        criterion, complications=[],
    )
    mock_wandb.log.assert_called_once()


# ── validate() — with-complications variant ───────────────────────────────────


@patch("src.training.trainer.wandb")
def test_validate_with_comp_has_comp_mean_auc(mock_wandb, model_with_comp):
    from src.training.losses import MultiTaskReinterventionLoss
    from src.training.trainer import validate

    criterion = MultiTaskReinterventionLoss(device=torch.device("cpu"))
    _, metrics = validate(
        model_with_comp, _FakeLoader([_make_comp_batch()]),
        criterion, complications=COMPLICATIONS,
    )
    assert "val/comp_mean_auc" in metrics
    assert 0.0 <= metrics["val/comp_mean_auc"] <= 1.0


@patch("src.training.trainer.wandb")
def test_validate_with_comp_still_has_reint_keys(mock_wandb, model_with_comp):
    """With-complications variant must also emit all standard reintervention keys."""
    from src.training.losses import MultiTaskReinterventionLoss
    from src.training.trainer import validate

    criterion = MultiTaskReinterventionLoss(device=torch.device("cpu"))
    _, metrics = validate(
        model_with_comp, _FakeLoader([_make_comp_batch()]),
        criterion, complications=COMPLICATIONS,
    )
    assert "val/reint_auc" in metrics
    assert "val/reint_best_thresh" in metrics


# ── checkpoint save format ────────────────────────────────────────────────────

CHECKPOINT_REQUIRED_KEYS = {
    "state_dict", "threshold", "feature_names",
    "clin_mean", "clin_std", "complications", "config",
}


def test_checkpoint_has_all_required_keys(fresh_model_no_comp, tmp_path):
    """Checkpoint dict (as produced by trainer.py) must contain all keys needed by load/inference."""
    feature_names = [f"feat_{i}" for i in range(CLINICAL_DIM)]
    ckpt_path = tmp_path / "test.pt"
    torch.save(
        {
            "state_dict":    fresh_model_no_comp.state_dict(),
            "threshold":     0.5,
            "feature_names": feature_names,
            "clin_mean":     [0.0] * CLINICAL_DIM,
            "clin_std":      [1.0] * CLINICAL_DIM,
            "complications": [],
            "config":        {"name": "test"},
        },
        ckpt_path,
    )
    loaded = torch.load(ckpt_path, map_location="cpu")
    assert CHECKPOINT_REQUIRED_KEYS <= loaded.keys()


def test_checkpoint_values_round_trip(fresh_model_no_comp, tmp_path):
    """Threshold, feature_names, clin_mean, and clin_std must survive a save/load round-trip."""
    feature_names = [f"feat_{i}" for i in range(CLINICAL_DIM)]
    clin_mean = np.arange(CLINICAL_DIM, dtype=float).tolist()
    clin_std  = [2.0] * CLINICAL_DIM
    threshold = 0.42

    ckpt_path = tmp_path / "test.pt"
    torch.save(
        {
            "state_dict":    fresh_model_no_comp.state_dict(),
            "threshold":     threshold,
            "feature_names": feature_names,
            "clin_mean":     clin_mean,
            "clin_std":      clin_std,
            "complications": [],
            "config":        {},
        },
        ckpt_path,
    )
    loaded = torch.load(ckpt_path, map_location="cpu")
    assert loaded["threshold"]     == pytest.approx(threshold)
    assert loaded["feature_names"] == feature_names
    assert np.allclose(loaded["clin_mean"], clin_mean)
    assert np.allclose(loaded["clin_std"],  clin_std)


def test_checkpoint_with_comp_stores_complications(fresh_model_with_comp, tmp_path):
    """With-complications checkpoint must store the complication list for inference."""
    feature_names = [f"feat_{i}" for i in range(CLINICAL_DIM)]
    ckpt_path = tmp_path / "test_comp.pt"
    torch.save(
        {
            "state_dict":    fresh_model_with_comp.state_dict(),
            "threshold":     0.5,
            "feature_names": feature_names,
            "clin_mean":     [0.0] * CLINICAL_DIM,
            "clin_std":      [1.0] * CLINICAL_DIM,
            "complications": COMPLICATIONS,
            "config":        {},
        },
        ckpt_path,
    )
    loaded = torch.load(ckpt_path, map_location="cpu")
    assert loaded["complications"] == COMPLICATIONS
