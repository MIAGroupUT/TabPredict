"""
Tests for src/evaluation/metrics.py

Covers:
- label_type: all four outcomes TP/FP/TN/FN
- compute_overall_metrics: perfect / all-wrong / random predictions
- compute_overall_metrics: with per-complication AUC
- compute_overall_metrics: skips comp AUC when too few unique labels
- save_metrics: writes JSON + CSV, CSV handles nested comp_auc_detail
- format_xai_report: returns a non-empty string containing patient info
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


# ── label_type ────────────────────────────────────────────────────────────────


def test_label_type_tp():
    from src.evaluation.metrics import label_type
    assert label_type(1, 1) == "TP"


def test_label_type_fp():
    from src.evaluation.metrics import label_type
    assert label_type(1, 0) == "FP"


def test_label_type_tn():
    from src.evaluation.metrics import label_type
    assert label_type(0, 0) == "TN"


def test_label_type_fn():
    from src.evaluation.metrics import label_type
    assert label_type(0, 1) == "FN"


# ── compute_overall_metrics — structure ───────────────────────────────────────


def _perfect_inputs(n=10):
    probs = [0.9] * (n // 2) + [0.1] * (n // 2)
    gt    = [1]   * (n // 2) + [0]   * (n // 2)
    return probs, gt


def test_compute_overall_metrics_required_keys():
    from src.evaluation.metrics import compute_overall_metrics
    probs, gt = _perfect_inputs()
    m = compute_overall_metrics(probs, gt, threshold=0.5)
    for key in ("n_samples", "auc", "f1", "precision", "recall", "accuracy",
                "threshold_used", "TP", "FP", "TN", "FN",
                "TPR_sensitivity", "TNR_specificity"):
        assert key in m, f"Missing key: {key}"


def test_compute_overall_metrics_n_samples():
    from src.evaluation.metrics import compute_overall_metrics
    probs, gt = _perfect_inputs(20)
    m = compute_overall_metrics(probs, gt, threshold=0.5)
    assert m["n_samples"] == 20


def test_compute_overall_metrics_n_positive():
    from src.evaluation.metrics import compute_overall_metrics
    probs, gt = _perfect_inputs(10)
    m = compute_overall_metrics(probs, gt, threshold=0.5)
    assert m["n_positive"] == 5
    assert m["n_negative"] == 5


# ── compute_overall_metrics — numeric correctness ─────────────────────────────


def test_compute_overall_metrics_perfect_auc():
    from src.evaluation.metrics import compute_overall_metrics
    probs, gt = _perfect_inputs()
    m = compute_overall_metrics(probs, gt, threshold=0.5)
    assert m["auc"] == pytest.approx(1.0)


def test_compute_overall_metrics_perfect_accuracy():
    from src.evaluation.metrics import compute_overall_metrics
    probs, gt = _perfect_inputs()
    m = compute_overall_metrics(probs, gt, threshold=0.5)
    assert m["accuracy"] == pytest.approx(1.0)


def test_compute_overall_metrics_all_wrong():
    from src.evaluation.metrics import compute_overall_metrics
    # All ones predicted as 0 and vice-versa
    n   = 10
    probs = [0.1] * (n // 2) + [0.9] * (n // 2)  # reversed
    gt    = [1]   * (n // 2) + [0]   * (n // 2)
    m = compute_overall_metrics(probs, gt, threshold=0.5)
    assert m["accuracy"] == pytest.approx(0.0)


def test_compute_overall_metrics_confusion_matrix_entries():
    from src.evaluation.metrics import compute_overall_metrics
    # 5 TP, 0 FP, 0 TN, 5 FN
    probs = [0.9] * 5 + [0.1] * 5
    gt    = [1]   * 5 + [1]   * 5   # all positive
    m     = compute_overall_metrics(probs, gt, threshold=0.5)
    assert m["TP"] == 5
    assert m["FN"] == 5
    assert m["FP"] == 0
    assert m["TN"] == 0


def test_compute_overall_metrics_threshold_used():
    from src.evaluation.metrics import compute_overall_metrics
    probs, gt = _perfect_inputs()
    m = compute_overall_metrics(probs, gt, threshold=0.7)
    assert m["threshold_used"] == pytest.approx(0.7, abs=0.001)


def test_compute_overall_metrics_auc_range():
    from src.evaluation.metrics import compute_overall_metrics
    probs = np.random.RandomState(0).rand(30).tolist()
    gt    = np.random.RandomState(1).randint(0, 2, 30).tolist()
    m     = compute_overall_metrics(probs, gt, threshold=0.5)
    assert 0.0 <= m["auc"] <= 1.0


def test_compute_overall_metrics_single_class_gt():
    """All-same GT: AUC undefined → returns 0.5."""
    from src.evaluation.metrics import compute_overall_metrics
    probs = [0.8] * 10
    gt    = [1]   * 10  # only positives
    m     = compute_overall_metrics(probs, gt, threshold=0.5)
    assert m["auc"] == 0.5


# ── compute_overall_metrics — with complications ──────────────────────────────


def test_compute_overall_metrics_comp_auc_present():
    from src.evaluation.metrics import compute_overall_metrics
    n     = 20
    probs, gt = _perfect_inputs(n)
    # Two complications with clear signal
    comp_probs = [np.array([0.9, 0.1])] * (n // 2) + [np.array([0.1, 0.9])] * (n // 2)
    comp_gt    = [np.array([1.0, 0.0])] * (n // 2) + [np.array([0.0, 1.0])] * (n // 2)
    m = compute_overall_metrics(probs, gt, 0.5, comp_probs, comp_gt, ["compA", "compB"])
    assert "comp_auc_detail" in m
    assert "comp_mean_auc"   in m


def test_compute_overall_metrics_comp_auc_values():
    from src.evaluation.metrics import compute_overall_metrics
    n = 20
    probs, gt  = _perfect_inputs(n)
    comp_probs = [np.array([0.9])] * (n // 2) + [np.array([0.1])] * (n // 2)
    comp_gt    = [np.array([1.0])] * (n // 2) + [np.array([0.0])] * (n // 2)
    m = compute_overall_metrics(probs, gt, 0.5, comp_probs, comp_gt, ["compA"])
    assert m["comp_auc_detail"]["compA"] == pytest.approx(1.0)


def test_compute_overall_metrics_comp_all_missing():
    """All comp GT == -1 → no comp_auc_detail key."""
    from src.evaluation.metrics import compute_overall_metrics
    n = 10
    probs, gt  = _perfect_inputs(n)
    comp_probs = [np.array([0.5])] * n
    comp_gt    = [np.array([-1.0])] * n
    m = compute_overall_metrics(probs, gt, 0.5, comp_probs, comp_gt, ["compA"])
    assert "comp_auc_detail" not in m


def test_compute_overall_metrics_no_comp_args():
    """Called without comp args → no comp keys in result."""
    from src.evaluation.metrics import compute_overall_metrics
    probs, gt = _perfect_inputs()
    m = compute_overall_metrics(probs, gt, 0.5)
    assert "comp_auc_detail" not in m


# ── save_metrics ──────────────────────────────────────────────────────────────


def test_save_metrics_creates_json(tmp_path):
    from src.evaluation.metrics import save_metrics
    m = {"auc": 0.85, "f1": 0.80, "n_samples": 10}
    save_metrics(m, tmp_path)
    assert (tmp_path / "test_metrics.json").exists()


def test_save_metrics_creates_csv(tmp_path):
    from src.evaluation.metrics import save_metrics
    m = {"auc": 0.85, "f1": 0.80}
    save_metrics(m, tmp_path)
    assert (tmp_path / "test_metrics.csv").exists()


def test_save_metrics_json_content(tmp_path):
    from src.evaluation.metrics import save_metrics
    m = {"auc": 0.85, "n_samples": 42}
    save_metrics(m, tmp_path)
    loaded = json.loads((tmp_path / "test_metrics.json").read_text())
    assert loaded["auc"]       == pytest.approx(0.85)
    assert loaded["n_samples"] == 42


def test_save_metrics_csv_has_header(tmp_path):
    from src.evaluation.metrics import save_metrics
    save_metrics({"auc": 0.9}, tmp_path)
    rows = list(csv.reader((tmp_path / "test_metrics.csv").open()))
    assert rows[0] == ["metric", "value"]


def test_save_metrics_csv_flattens_comp_auc_detail(tmp_path):
    from src.evaluation.metrics import save_metrics
    m = {"auc": 0.9, "comp_auc_detail": {"endoleak_1a": 0.75, "migration": 0.80}}
    save_metrics(m, tmp_path)
    rows = list(csv.reader((tmp_path / "test_metrics.csv").open()))
    flat_keys = [r[0] for r in rows[1:]]
    assert "auc_endoleak_1a" in flat_keys
    assert "auc_migration"   in flat_keys
    assert "comp_auc_detail" not in flat_keys


# ── format_xai_report ─────────────────────────────────────────────────────────


def _make_xai_dict(with_comp=False):
    d = {
        "reintervention_prob":       0.72,
        "reintervention_confidence": 0.80,
        "confidence_level":          "HIGH",
        "clinical_importance":       [("feature_a", 0.05, "HIGH"), ("feature_b", 0.03, "MED")],
        "aaa_importance":            [("volume_lumen_ml", 0.8, "HIGH")],
        "modality_importance":       {"CTA": 0.6, "Clinical": 0.3, "AAA_metrics": 0.1},
        "causal_chain":              "Elevated feature_a → likely reintervention",
        "primary_risk_factor":       "feature_a (HIGH impact)",
    }
    if with_comp:
        d["complication_probs"]      = {"endoleak_1a": 0.65, "migration": 0.20}
        d["complication_confidence"] = {"endoleak_1a": "HIGH", "migration": "LOW"}
    return d


def _make_pred_info():
    return {
        "reintervention_prob": 0.72,
        "reintervention_pred": 1,
        "reintervention_gt":   1,
        "label_type":          "TP",
        "threshold":           0.5,
    }


def test_format_xai_report_returns_string():
    from src.evaluation.metrics import format_xai_report
    report = format_xai_report("patient_001", _make_pred_info(), _make_xai_dict())
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_xai_report_contains_pid():
    from src.evaluation.metrics import format_xai_report
    report = format_xai_report("patient_001", _make_pred_info(), _make_xai_dict())
    assert "patient_001" in report


def test_format_xai_report_contains_probability():
    from src.evaluation.metrics import format_xai_report
    report = format_xai_report("patient_42", _make_pred_info(), _make_xai_dict())
    assert "72" in report  # 72% probability


def test_format_xai_report_contains_clinical_features():
    from src.evaluation.metrics import format_xai_report
    report = format_xai_report("pid", _make_pred_info(), _make_xai_dict())
    assert "feature_a" in report


def test_format_xai_report_with_complications():
    from src.evaluation.metrics import format_xai_report
    report = format_xai_report(
        "pid", _make_pred_info(), _make_xai_dict(with_comp=True),
        complications=["endoleak_1a", "migration"],
    )
    assert "endoleak_1a" in report
    assert "migration"   in report


def test_format_xai_report_without_complications():
    from src.evaluation.metrics import format_xai_report
    report = format_xai_report("pid", _make_pred_info(), _make_xai_dict())
    assert "COMPLICATION" not in report


def test_format_xai_report_contains_label_type():
    from src.evaluation.metrics import format_xai_report
    pred = _make_pred_info()
    pred["label_type"] = "FP"
    report = format_xai_report("pid", pred, _make_xai_dict())
    assert "FP" in report
