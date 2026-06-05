"""
Tests for TabularBaseline (src/models/baselines.py) and
build_tabular_arrays (src/data/loaders.py).

All tests run on CPU with synthetic data — no GPU, no real files needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import AAA_DIM, CLINICAL_DIM

N_TRAIN = 20
N_VAL   = 8
N_FEAT  = CLINICAL_DIM + AAA_DIM   # total tabular width (clinical + AAA)

# ── synthetic data fixtures ───────────────────────────────────────────────────


@pytest.fixture
def tabular_data():
    """Small balanced train/val arrays with some NaN entries."""
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((N_TRAIN, N_FEAT)).astype(np.float32)
    X_va = rng.standard_normal((N_VAL,   N_FEAT)).astype(np.float32)
    # Introduce a few NaNs (missing clinical values)
    X_tr[rng.integers(0, N_TRAIN, 4), rng.integers(0, N_FEAT, 4)] = np.nan
    X_va[rng.integers(0, N_VAL,   2), rng.integers(0, N_FEAT, 2)] = np.nan
    y_tr = np.array([0, 1] * (N_TRAIN // 2), dtype=np.float32)
    y_va = np.array([0, 1] * (N_VAL   // 2), dtype=np.float32)
    return X_tr, y_tr, X_va, y_va


# ── build_tabular_arrays ──────────────────────────────────────────────────────


def _make_synthetic_config(n: int, seed: int = 42) -> tuple[list[str], dict]:
    """Build a minimal config + id list matching the loaders.py format."""
    rng    = np.random.default_rng(seed)
    ids    = [f"P{i:04d}" for i in range(n)]
    feats  = [f"feat_{j}" for j in range(CLINICAL_DIM)]
    config = {}
    for pid in ids:
        clin = {k: float(rng.standard_normal()) for k in feats}
        # Inject one NaN per patient
        clin[feats[0]] = float("nan")
        config[pid] = {
            "pre_op_features": clin,
            "scans":           {"pre_op": {"metrics": None}},
            "ground_truth":    {"reintervention": int(rng.integers(0, 2))},
        }
    return ids, feats, config


def test_build_tabular_arrays_shape():
    from src.data.loaders import build_tabular_arrays
    ids, feats, config = _make_synthetic_config(12)
    X, y, pids = build_tabular_arrays(ids, config, feats)
    assert X.shape == (12, CLINICAL_DIM + 5), "X width must be clinical_dim + 5 AAA scalars"
    assert y.shape == (12,)
    assert len(pids) == 12


def test_build_tabular_arrays_preserves_nan():
    """NaN values in clinical features must survive into X (callers handle imputation)."""
    from src.data.loaders import build_tabular_arrays
    ids, feats, config = _make_synthetic_config(10)
    X, _, _ = build_tabular_arrays(ids, config, feats)
    assert np.isnan(X[:, 0]).all(), "feat_0 was NaN for every patient"


def test_build_tabular_arrays_labels_binary():
    from src.data.loaders import build_tabular_arrays
    ids, feats, config = _make_synthetic_config(20)
    _, y, _ = build_tabular_arrays(ids, config, feats)
    assert set(y.astype(int).tolist()).issubset({0, 1})


def test_build_tabular_arrays_pid_order():
    from src.data.loaders import build_tabular_arrays
    ids, feats, config = _make_synthetic_config(6)
    _, _, pids = build_tabular_arrays(ids, config, feats)
    assert pids == ids


# ── TabularBaseline — common behaviour across all available models ────────────

from src.models.baselines import AVAILABLE, _XGB_AVAILABLE

_MODELS_TO_TEST = [m for m in AVAILABLE if m != "xgboost" or _XGB_AVAILABLE]


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_fit_predict_shape(model_type, tabular_data):
    from src.models.baselines import TabularBaseline
    X_tr, y_tr, X_va, _ = tabular_data
    bl = TabularBaseline(model_type)
    bl.fit(X_tr, y_tr)
    probs = bl.predict_proba(X_va)
    assert probs.shape == (N_VAL,), "predict_proba must return 1-D array of length N"


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_probs_in_unit_interval(model_type, tabular_data):
    from src.models.baselines import TabularBaseline
    X_tr, y_tr, X_va, _ = tabular_data
    bl = TabularBaseline(model_type)
    bl.fit(X_tr, y_tr)
    probs = bl.predict_proba(X_va)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_no_nan_in_output(model_type, tabular_data):
    """NaN inputs must not propagate to output probabilities."""
    from src.models.baselines import TabularBaseline
    X_tr, y_tr, X_va, _ = tabular_data
    bl = TabularBaseline(model_type)
    bl.fit(X_tr, y_tr)
    probs = bl.predict_proba(X_va)
    assert not np.isnan(probs).any()


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_save_load_roundtrip(model_type, tabular_data, tmp_path):
    from src.models.baselines import TabularBaseline
    X_tr, y_tr, X_va, _ = tabular_data
    bl = TabularBaseline(model_type)
    bl.fit(X_tr, y_tr)
    probs_before = bl.predict_proba(X_va)

    path = tmp_path / f"{model_type}.pkl"
    bl.save(path)
    bl2 = TabularBaseline.load(path)
    probs_after = bl2.predict_proba(X_va)

    np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5)


# ── TabularBaseline — guard: predict before fit raises ───────────────────────


def test_predict_before_fit_raises(tabular_data):
    from src.models.baselines import TabularBaseline
    X_tr, _, X_va, _ = tabular_data
    bl = TabularBaseline("lr")
    with pytest.raises(RuntimeError, match="fit()"):
        bl.predict_proba(X_va)


# ── xgboost unavailability is handled gracefully ─────────────────────────────


def test_xgboost_import_error_if_unavailable():
    if _XGB_AVAILABLE:
        pytest.skip("xgboost is installed — skip unavailability test")
    from src.models.baselines import TabularBaseline
    with pytest.raises(ImportError, match="xgboost"):
        TabularBaseline("xgboost")


# ── unknown model_type raises immediately ────────────────────────────────────


def test_unknown_model_type_raises():
    from src.models.baselines import TabularBaseline
    with pytest.raises(ValueError, match="Unknown model_type"):
        TabularBaseline("neural_ode")


# ── ClassicalBranchWrapper ────────────────────────────────────────────────────

import torch

@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_wrapper_returns_zeros_before_fit(model_type):
    from src.models.baselines import ClassicalBranchWrapper
    wrapper = ClassicalBranchWrapper(in_features=N_FEAT, model_type=model_type)
    x = torch.rand(4, N_FEAT)
    out = wrapper(x)
    assert out.shape == (4, 32)
    assert out.sum().item() == pytest.approx(0.0)


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_wrapper_forward_after_fit(model_type, tabular_data):
    from src.models.baselines import ClassicalBranchWrapper
    X_tr, y_tr, X_va, _ = tabular_data
    wrapper = ClassicalBranchWrapper(in_features=N_FEAT, model_type=model_type)
    wrapper.fit(X_tr, y_tr)
    x = torch.as_tensor(X_va)
    out = wrapper(x)
    assert out.shape == (N_VAL, 32)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_wrapper_projection_receives_gradients(model_type, tabular_data):
    """Gradients must flow through the Linear(1→32) projection layer."""
    from src.models.baselines import ClassicalBranchWrapper
    X_tr, y_tr, X_va, _ = tabular_data
    wrapper = ClassicalBranchWrapper(in_features=N_FEAT, model_type=model_type)
    wrapper.fit(X_tr, y_tr)
    x   = torch.as_tensor(X_va)
    out = wrapper(x)
    loss = out.sum()
    loss.backward()
    assert wrapper.projection.weight.grad is not None
    assert wrapper.projection.weight.grad.abs().sum().item() > 0


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_wrapper_serialize_deserialize(model_type, tabular_data, tmp_path):
    from src.models.baselines import ClassicalBranchWrapper
    X_tr, y_tr, X_va, _ = tabular_data
    wrapper = ClassicalBranchWrapper(in_features=N_FEAT, model_type=model_type)
    wrapper.fit(X_tr, y_tr)
    probs_before = wrapper.baseline.predict_proba(X_va)

    data = wrapper.serialize()
    baseline2 = ClassicalBranchWrapper.deserialize(data)
    probs_after = baseline2.predict_proba(X_va)

    np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5)


# ── is_classical_backbone helper ──────────────────────────────────────────────

def test_is_classical_backbone_false_for_mlp():
    import sys
    from unittest.mock import MagicMock
    import torch.nn as nn
    # Ensure merlin mock is present (conftest handles this at collection time)
    from src.models.reintervention import ReinterventionRiskModel, is_classical_backbone
    model = ReinterventionRiskModel(
        clinical_dim=N_FEAT, aaa_dim=0, clinical_backbone="mlp"
    )
    assert not is_classical_backbone(model)


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_is_classical_backbone_true_for_classical(model_type):
    from src.models.reintervention import ReinterventionRiskModel, is_classical_backbone
    model = ReinterventionRiskModel(
        clinical_dim=N_FEAT, aaa_dim=0, clinical_backbone=model_type
    )
    assert is_classical_backbone(model)


# ── fit_classical_backbone integration ───────────────────────────────────────

@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_fit_classical_backbone_marks_fitted(model_type):
    from src.models.reintervention import ReinterventionRiskModel
    from src.training.trainer import fit_classical_backbone

    ids, feats, config = _make_synthetic_config(20)
    mean = np.zeros(len(feats), dtype=np.float32)
    std  = np.ones(len(feats),  dtype=np.float32)

    model = ReinterventionRiskModel(
        clinical_dim=len(feats), aaa_dim=5, clinical_backbone=model_type
    )
    fit_classical_backbone(model, ids, config, feats, mean, std)
    assert model.clinical_branch._fitted is True


@pytest.mark.parametrize("model_type", _MODELS_TO_TEST)
def test_fit_classical_backbone_forward_no_zeros(model_type):
    """After fitting, forward pass must return non-zero output."""
    from src.models.reintervention import ReinterventionRiskModel
    from src.training.trainer import fit_classical_backbone

    ids, feats, config = _make_synthetic_config(20)
    mean = np.zeros(len(feats), dtype=np.float32)
    std  = np.ones(len(feats),  dtype=np.float32)

    model = ReinterventionRiskModel(
        clinical_dim=len(feats), aaa_dim=5, clinical_backbone=model_type
    )
    fit_classical_backbone(model, ids, config, feats, mean, std)

    clin = torch.rand(2, len(feats))
    aaa  = torch.rand(2, 5)
    out  = model.clinical_branch(torch.cat([clin, aaa], dim=1))
    assert out.shape == (2, 32)
    # After fitting projection weights are random-init, output won't be all zeros
    assert not torch.all(out == 0)


# ── TabPFN backbone — structural tests (no fitting; weights download is slow) ─

def test_tabpfn_in_classical_backbones():
    from src.models.baselines import CLASSICAL_BACKBONES
    assert "tabpfn" in CLASSICAL_BACKBONES


def test_tabpfn_not_in_available():
    """AVAILABLE is the sklearn-only list used for TabularBaseline parametrisation."""
    from src.models.baselines import AVAILABLE
    assert "tabpfn" not in AVAILABLE


def test_tabpfn_baseline_builds():
    from src.models.baselines import TabPFNBaseline
    b = TabPFNBaseline(epochs=1, n_estimators_final_inference=1)
    assert b is not None
    assert not b._fitted


def test_classical_branch_wrapper_selects_tabpfn_baseline():
    from src.models.baselines import ClassicalBranchWrapper, TabPFNBaseline
    w = ClassicalBranchWrapper(in_features=N_FEAT, model_type="tabpfn",
                               epochs=1, n_estimators_final_inference=1)
    assert isinstance(w.baseline, TabPFNBaseline)


def test_tabpfn_wrapper_returns_zeros_before_fit():
    from src.models.baselines import ClassicalBranchWrapper
    w = ClassicalBranchWrapper(in_features=N_FEAT, model_type="tabpfn",
                               epochs=1, n_estimators_final_inference=1)
    x = torch.rand(4, N_FEAT)
    out = w(x)
    assert out.shape == (4, 32)
    assert out.sum().item() == pytest.approx(0.0)


def test_is_classical_backbone_true_for_tabpfn():
    from src.models.reintervention import ReinterventionRiskModel, is_classical_backbone
    model = ReinterventionRiskModel(
        clinical_dim=N_FEAT, aaa_dim=0, clinical_backbone="tabpfn",
        tabm_cfg=dict(epochs=1, n_estimators_final_inference=1),
    )
    assert is_classical_backbone(model)
