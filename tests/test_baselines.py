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
    """Gradients must flow through the projection layer."""
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


# ── projection width per model type ──────────────────────────────────────────

_PROJ_IN_EXPECTED = [
    ("lr",      1),
    ("rf",      300),
    ("gbm",     300),
    ("xgboost", N_FEAT),
]

@pytest.mark.parametrize("model_type,expected_proj_in", [
    (m, p) for m, p in _PROJ_IN_EXPECTED if m != "xgboost" or _XGB_AVAILABLE
])
def test_wrapper_projection_width(model_type, expected_proj_in):
    """Projection input width must match the per-model representation size."""
    from src.models.baselines import ClassicalBranchWrapper
    wrapper = ClassicalBranchWrapper(in_features=N_FEAT, model_type=model_type)
    assert wrapper.projection.in_features == expected_proj_in
    assert wrapper.projection.out_features == 32


def test_gbm_leaf_scale_stored_after_fit(tabular_data):
    """After fit(), GBM leaf scale must be stored on baseline and have shape (n_estimators,)."""
    from src.models.baselines import ClassicalBranchWrapper
    X_tr, y_tr, _, _ = tabular_data
    wrapper = ClassicalBranchWrapper(in_features=N_FEAT, model_type='gbm')
    wrapper.fit(X_tr, y_tr)
    assert hasattr(wrapper.baseline, '_gbm_leaf_scale')
    scale = wrapper.baseline._gbm_leaf_scale
    assert scale.shape == (300,)
    assert (scale >= 1.0).all()


@pytest.mark.parametrize("model_type,expected_cols", [
    ("lr",  1),
    ("rf",  300),
    ("gbm", 300),
])
def test_extract_features_shape(model_type, expected_cols, tabular_data):
    """_extract_features must return (B, expected_cols) float32 array."""
    from src.models.baselines import ClassicalBranchWrapper
    X_tr, y_tr, X_va, _ = tabular_data
    wrapper = ClassicalBranchWrapper(in_features=N_FEAT, model_type=model_type)
    wrapper.fit(X_tr, y_tr)
    feat = wrapper._extract_features(X_va)
    assert feat.shape == (N_VAL, expected_cols)
    assert feat.dtype == np.float32


# ── imputation behaviour per model type ──────────────────────────────────────


def test_xgboost_impute_preserves_nans():
    """XGBoost skips _impute so NaNs reach the model's native handler."""
    from src.models.baselines import TabularBaseline
    baseline = TabularBaseline("xgboost")
    X = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    result = baseline._impute(X, fit=True)
    assert np.isnan(result[0, 1]), "XGBoost should receive NaN, not a filled value"


@pytest.mark.parametrize("model_type", ["lr", "rf", "gbm"])
def test_non_xgboost_impute_fills_nans_with_median(model_type):
    """LR / RF / GBM: _impute fills NaNs with the training-set column median."""
    from src.models.baselines import TabularBaseline
    baseline = TabularBaseline(model_type)
    # col 0: values 1, 3 → median 2.0; col 1: value 5 only (NaN excluded) → median 5.0
    X_fit = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    result = baseline._impute(X_fit, fit=True)
    assert result[0, 1] == pytest.approx(5.0)   # NaN filled with median of col 1
    assert not np.isnan(result).any()


def test_fit_classical_backbone_passes_nans_to_branch(monkeypatch):
    """fit_classical_backbone must NOT zero NaNs before calling branch.fit()."""
    from src.models.reintervention import ReinterventionRiskModel
    from src.training.trainer import fit_classical_backbone

    ids, feats, config = _make_synthetic_config(20)
    # Inject a NaN into every patient's first feature
    for pid in ids:
        config[pid]["pre_op_features"][feats[0]] = float("nan")

    mean = np.zeros(len(feats), dtype=np.float32)
    std  = np.ones(len(feats),  dtype=np.float32)

    received: list = []

    model = ReinterventionRiskModel(
        clinical_dim=len(feats), aaa_dim=5, clinical_backbone="lr"
    )
    original_fit = model.clinical_branch.baseline.fit

    def capturing_fit(X, y):
        received.append(X.copy())
        return original_fit(X, y)

    monkeypatch.setattr(model.clinical_branch.baseline, "fit", capturing_fit)
    fit_classical_backbone(model, ids, config, feats, mean, std)

    assert len(received) == 1
    # First feature column must still contain NaN — not pre-zeroed
    assert np.isnan(received[0][:, 0]).any(), (
        "fit_classical_backbone should not zero NaNs before calling branch.fit()"
    )


# TabPFN tests live in tests/test_tabpfn.py
