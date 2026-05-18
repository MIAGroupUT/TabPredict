"""
Metric computation and human-readable reporting for test-set evaluation.
No W&B dependency — purely local outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_overall_metrics(
    all_probs: list[float],
    all_gt: list[int],
    threshold: float,
    all_comp_probs: list[np.ndarray] | None = None,
    all_comp_gt: list[np.ndarray] | None = None,
    complications: list[str] | None = None,
) -> dict:
    """
    Compute all metrics tracked during training, evaluated on the test split.

    Returns a flat (mostly) JSON-serialisable dict.
    per-complication AUC is nested under ``comp_auc_detail`` when available.
    """
    probs = np.array(all_probs)
    gt    = np.array(all_gt).astype(int)
    preds = (probs > threshold).astype(int)

    p_c, r_c, t_c = precision_recall_curve(gt, probs)
    f1s           = 2 * p_c[:-1] * r_c[:-1] / (p_c[:-1] + r_c[:-1] + 1e-8)
    best_idx      = int(np.argmax(f1s))

    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    metrics: dict = {
        "n_samples":       int(len(gt)),
        "n_positive":      int(gt.sum()),
        "n_negative":      int((~gt.astype(bool)).sum()),
        "threshold_used":  round(float(threshold), 4),
        "auc":             round(float(roc_auc_score(gt, probs)), 4)
                           if len(np.unique(gt)) > 1 else 0.5,
        "f1":              round(float(f1_score(gt, preds)), 4),
        "precision":       round(float(precision_score(gt, preds, zero_division=0)), 4),
        "recall":          round(float(recall_score(gt, preds, zero_division=0)), 4),
        "accuracy":        round(float(accuracy_score(gt, preds)), 4),
        "best_f1":         round(float(f1s[best_idx]), 4),
        "best_threshold":  round(float(t_c[best_idx]), 4),
        "TP":              int(tp),
        "FP":              int(fp),
        "TN":              int(tn),
        "FN":              int(fn),
        "TPR_sensitivity": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
        "TNR_specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        "FPR_fallout":     round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
        "FNR_missrate":    round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4),
    }

    if all_comp_probs and all_comp_gt and complications:
        comp_p    = np.stack(all_comp_probs)   # (N, C)
        comp_g    = np.stack(all_comp_gt)      # (N, C)
        comp_aucs = {}
        for i, name in enumerate(complications):
            valid = comp_g[:, i] != -1
            if valid.sum() < 2:
                continue
            y_t = comp_g[valid, i]
            y_p = comp_p[valid, i]
            if len(np.unique(y_t)) > 1:
                comp_aucs[name] = round(float(roc_auc_score(y_t, y_p)), 4)
        if comp_aucs:
            metrics["comp_mean_auc"]   = round(float(np.mean(list(comp_aucs.values()))), 4)
            metrics["comp_auc_detail"] = comp_aucs

    return metrics


def save_metrics(metrics: dict, out_dir: Path) -> None:
    """Write test_metrics.json and test_metrics.csv to out_dir."""
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "test_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            if k == "comp_auc_detail":
                for comp, auc in v.items():
                    writer.writerow([f"auc_{comp}", auc])
            else:
                writer.writerow([k, v])


def label_type(pred: int, gt: int) -> str:
    return {(1, 1): "TP", (1, 0): "FP", (0, 0): "TN"}.get((pred, gt), "FN")


def format_xai_report(
    pid: str,
    pred_info: dict,
    xai: dict,
    complications: list[str] | None = None,
) -> str:
    """Human-readable per-patient report."""
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  PATIENT: {pid:<52}║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        "PREDICTION:",
        f"  Reintervention probability : {pred_info['reintervention_prob']:.1%}",
        f"  Prediction (at threshold)  : {'POSITIVE' if pred_info['reintervention_pred'] else 'NEGATIVE'}",
        f"  Ground truth               : {'POSITIVE' if pred_info['reintervention_gt'] else 'NEGATIVE'}",
        f"  Label type                 : {pred_info['label_type']}",
        f"  Threshold used             : {pred_info['threshold']:.4f}",
        f"  Confidence                 : {xai['confidence_level']} ({xai['reintervention_confidence']:.2f})",
    ]

    if complications and "complication_probs" in xai:
        lines += ["", "COMPLICATION BREAKDOWN (sorted by probability):"]
        for comp, prob in sorted(
            xai["complication_probs"].items(), key=lambda x: x[1], reverse=True
        ):
            conf = xai.get("complication_confidence", {}).get(comp, "?")
            lines.append(f"  • {comp:<22}  {prob:.1%}  [{conf} confidence]")

    lines += ["", "TOP CLINICAL FACTORS (gradient-based, top 10):"]
    for feat, score, level in xai.get("clinical_importance", [])[:10]:
        lines.append(f"  • {feat:<44} {level:4} ({score:.4f})")

    lines += ["", "AAA MORPHOLOGY IMPORTANCE:"]
    for metric, score, level in xai.get("aaa_importance", []):
        lines.append(f"  • {metric:<44} {level:4} ({score:.4f})")

    lines += ["", "MODALITY IMPORTANCE (ablation):"]
    for mod, imp in sorted(
        xai.get("modality_importance", {}).items(), key=lambda x: x[1], reverse=True
    ):
        lines.append(f"  • {mod:<22} {imp:.1%}")

    lines += [
        "",
        "CAUSAL CHAIN:",
        f"  {xai.get('causal_chain', 'N/A')}",
        "",
        "PRIMARY RISK FACTOR:",
        f"  {xai.get('primary_risk_factor', 'N/A')}",
        "",
    ]
    return "\n".join(lines)
