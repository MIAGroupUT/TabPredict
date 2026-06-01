"""
Tests for src/xai/explainer.py

All tests run on CPU with the mock Merlin model.

Covers:
- ReinterventionXAI.__init__: stores feature_names/complications correctly
- _clinical_importance: returns list of (name, score, level) tuples
- _aaa_importance: returns all 5 AAA metric entries
- _modality_importance: returns dict with CTA/Clinical/AAA_metrics keys summing to 1
- full_explain (no-comp): all expected keys present, numeric ranges valid
- full_explain (with-comp): complication_probs and confidence keys present
- gradcam: returns None when no I3ResNet layer (mock backbone)
- _conf_level: correct thresholding
- _get_i3res: returns None for mock model
- save_gradcam_pngs: creates PNG files on disk
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import (
    BATCH, CLINICAL_DIM, AAA_DIM, SPATIAL, COMPLICATIONS, N_COMP,
)

FEATURE_NAMES = [f"feat_{i}" for i in range(CLINICAL_DIM)]


@pytest.fixture(scope="module")
def xai_no_comp(model_no_comp):
    from src.xai.explainer import ReinterventionXAI
    return ReinterventionXAI(model_no_comp, FEATURE_NAMES)


@pytest.fixture(scope="module")
def xai_with_comp(model_with_comp):
    from src.xai.explainer import ReinterventionXAI
    return ReinterventionXAI(model_with_comp, FEATURE_NAMES, complications=COMPLICATIONS)


# ── __init__ ──────────────────────────────────────────────────────────────────


def test_xai_stores_feature_names(xai_no_comp):
    assert len(xai_no_comp.feature_names) == CLINICAL_DIM


def test_xai_stores_complications(xai_with_comp):
    assert xai_with_comp.complications == COMPLICATIONS


def test_xai_no_comp_empty_complications(xai_no_comp):
    assert xai_no_comp.complications == []


# ── _conf_level ───────────────────────────────────────────────────────────────


def test_conf_level_high(xai_no_comp):
    # conf = 2*|0.9 - 0.5| = 0.8 > 0.6 → HIGH
    assert xai_no_comp._conf_level(0.8) == "HIGH"


def test_conf_level_med(xai_no_comp):
    # conf = 2*|0.7 - 0.5| = 0.4, 0.3 < 0.4 <= 0.6 → MED
    assert xai_no_comp._conf_level(0.4) == "MED"


def test_conf_level_low(xai_no_comp):
    # conf = 2*|0.55 - 0.5| = 0.1 <= 0.3 → LOW
    assert xai_no_comp._conf_level(0.1) == "LOW"


def test_conf_level_boundary_high(xai_no_comp):
    # 0.61 > 0.6 → HIGH
    assert xai_no_comp._conf_level(0.61) == "HIGH"


def test_conf_level_boundary_med(xai_no_comp):
    # 0.31 > 0.3 → MED
    assert xai_no_comp._conf_level(0.31) == "MED"


# ── _get_i3res ────────────────────────────────────────────────────────────────


def test_get_i3res_returns_none_for_mock(xai_no_comp):
    """Mock Merlin has no layer4/layer1 so _get_i3res must return None."""
    assert xai_no_comp._get_i3res() is None


# ── _clinical_importance ──────────────────────────────────────────────────────


def test_clinical_importance_returns_list(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._clinical_importance(cta[:1], clin[:1], aaa[:1])
    assert isinstance(result, list)


def test_clinical_importance_tuple_structure(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._clinical_importance(cta[:1], clin[:1], aaa[:1])
    if result:  # may be empty if grad is None
        name, score, level = result[0]
        assert isinstance(name, str)
        assert isinstance(score, float)
        assert level in ("HIGH", "MED", "LOW")


def test_clinical_importance_max_20_entries(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._clinical_importance(cta[:1], clin[:1], aaa[:1])
    assert len(result) <= 20


def test_clinical_importance_feature_names_valid(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._clinical_importance(cta[:1], clin[:1], aaa[:1])
    valid  = set(FEATURE_NAMES)
    for name, _, _ in result:
        assert name in valid


# ── _aaa_importance ───────────────────────────────────────────────────────────


def test_aaa_importance_returns_5_entries(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._aaa_importance(cta[:1], clin[:1], aaa[:1])
    assert len(result) == 5


def test_aaa_importance_tuple_structure(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._aaa_importance(cta[:1], clin[:1], aaa[:1])
    for name, score, level in result:
        assert isinstance(name, str)
        assert 0.0 <= score <= 1.0 + 1e-5
        assert level in ("HIGH", "MED", "LOW")


def test_aaa_importance_known_metric_names(xai_no_comp, cta, clin, aaa):
    from src.xai.explainer import AAA_METRIC_NAMES
    result = xai_no_comp._aaa_importance(cta[:1], clin[:1], aaa[:1])
    names  = [r[0] for r in result]
    for n in AAA_METRIC_NAMES:
        assert n in names


# ── _modality_importance ──────────────────────────────────────────────────────


def test_modality_importance_keys(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._modality_importance(cta[:1], clin[:1], aaa[:1])
    assert set(result.keys()) == {"CTA", "Clinical", "AAA_metrics"}


def test_modality_importance_sums_to_one(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._modality_importance(cta[:1], clin[:1], aaa[:1])
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-4)


def test_modality_importance_non_negative(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp._modality_importance(cta[:1], clin[:1], aaa[:1])
    assert all(v >= 0 for v in result.values())


# ── full_explain — no complications ──────────────────────────────────────────


EXPECTED_KEYS_BASE = {
    "reintervention_prob", "reintervention_confidence", "confidence_level",
    "clinical_importance", "aaa_importance", "modality_importance",
    "causal_chain", "primary_risk_factor",
}


def test_full_explain_no_comp_keys(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    assert EXPECTED_KEYS_BASE.issubset(result.keys())


def test_full_explain_no_comp_prob_range(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    assert 0.0 <= result["reintervention_prob"] <= 1.0


def test_full_explain_no_comp_confidence_range(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    assert 0.0 <= result["reintervention_confidence"] <= 1.0


def test_full_explain_no_comp_confidence_level(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    assert result["confidence_level"] in ("HIGH", "MED", "LOW")


def test_full_explain_no_comp_no_complication_keys(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    assert "complication_probs"      not in result
    assert "complication_confidence" not in result


def test_full_explain_no_comp_causal_chain_string(xai_no_comp, cta, clin, aaa):
    result = xai_no_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    assert isinstance(result["causal_chain"], str)
    assert len(result["causal_chain"]) > 0


# ── full_explain — with complications ────────────────────────────────────────


def test_full_explain_with_comp_keys(xai_with_comp, cta, clin, aaa):
    result = xai_with_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    assert "complication_probs"      in result
    assert "complication_confidence" in result


def test_full_explain_with_comp_names(xai_with_comp, cta, clin, aaa):
    result = xai_with_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    assert set(result["complication_probs"].keys())      == set(COMPLICATIONS)
    assert set(result["complication_confidence"].keys()) == set(COMPLICATIONS)


def test_full_explain_with_comp_prob_range(xai_with_comp, cta, clin, aaa):
    result = xai_with_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    for p in result["complication_probs"].values():
        assert 0.0 <= p <= 1.0


def test_full_explain_with_comp_confidence_levels(xai_with_comp, cta, clin, aaa):
    result = xai_with_comp.full_explain(cta[:1], clin[:1], aaa[:1])
    for lvl in result["complication_confidence"].values():
        assert lvl in ("HIGH", "MED", "LOW")


# ── gradcam ───────────────────────────────────────────────────────────────────


def test_gradcam_returns_none_with_mock_backbone(xai_no_comp, cta, clin, aaa):
    """Mock Merlin has no I3ResNet → gradcam must return None gracefully."""
    result = xai_no_comp.gradcam(cta[:1], clin[:1], aaa[:1])
    assert result is None


def test_gradcam_with_comp_returns_none(xai_with_comp, cta, clin, aaa):
    result = xai_with_comp.gradcam(cta[:1], clin[:1], aaa[:1])
    assert result is None


# ── save_gradcam_pngs ─────────────────────────────────────────────────────────


def test_save_gradcam_pngs_creates_files(xai_no_comp, cta, tmp_path):
    D = cta.shape[2]
    cam_np = np.random.rand(D, D, D).astype(np.float32)
    xai_no_comp.save_gradcam_pngs(cta[:1], cam_np, tmp_path)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) >= 4  # combined + 3 individual views


def test_save_gradcam_pngs_names(xai_no_comp, cta, tmp_path):
    D = cta.shape[2]
    cam_np = np.zeros((D, D, D), dtype=np.float32)
    xai_no_comp.save_gradcam_pngs(cta[:1], cam_np, tmp_path)
    names = {p.name for p in tmp_path.glob("*.png")}
    assert "gradcam_combined.png" in names
    assert "gradcam_axial.png"    in names
    assert "gradcam_sagittal.png" in names
    assert "gradcam_coronal.png"  in names


# ── _render_slice (static method) ────────────────────────────────────────────


def test_render_slice_runs(xai_no_comp):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sl  = np.zeros((6, 6))
    cam = np.zeros((6, 6))
    lumen = np.zeros((6, 6), dtype=bool)
    ilt   = np.zeros((6, 6), dtype=bool)
    xai_no_comp._render_slice(ax, sl, cam, lumen, ilt, "test")
    plt.close(fig)
