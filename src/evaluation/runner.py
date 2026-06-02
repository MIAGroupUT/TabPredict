"""
Test-set evaluation runner.

Entry point: run_test(cfg, checkpoint_path, output_dir, gradcam, split)

Produces, for each patient:
    <out_dir>/<pid>/prediction.json
    <out_dir>/<pid>/xai.json
    <out_dir>/<pid>/xai_report.txt
    <out_dir>/<pid>/gradcam_*.png          (only with gradcam=True)

And at the parent level:
    <out_dir>/test_metrics.json
    <out_dir>/test_metrics.csv
    <out_dir>/test_log.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset
from torch.utils.data import DataLoader

from src.data.loaders import (
    build_data_list,
    get_cached_transforms,
    get_val_runtime_transforms,
    load_splits_and_config,
    RuntimeDataset,
)
from src.evaluation.metrics import (
    compute_overall_metrics,
    format_xai_report,
    label_type,
    save_metrics,
)
from src.models.reintervention import ReinterventionRiskModel, load_checkpoint
from src.xai.explainer import ReinterventionXAI

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_test(
    cfg: dict,
    checkpoint_path: str,
    output_dir: str,
    gradcam: bool = False,
    split: str = "test",
    num_workers: int = 8,
) -> None:
    """
    Evaluate a trained checkpoint on the requested data split.

    Args:
        cfg:             Experiment config dict (from experiments/*.yaml).
        checkpoint_path: Path to a .pt checkpoint.
        output_dir:      Root directory for all outputs.
        gradcam:         Compute and save GradCAM PNGs (slower).
        split:           One of "train" / "val" / "test".
        num_workers:     DataLoader workers.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "test_log.txt", "w")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Config     : {cfg.get('name', '?')}")
    log(f"Checkpoint : {checkpoint_path}")
    log(f"Output dir : {out_dir}")
    log(f"Split      : {split}")
    log(f"GradCAM    : {gradcam}")
    log(f"Device     : {DEVICE}")

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt          = torch.load(checkpoint_path, map_location=DEVICE)
    feature_names = ckpt["feature_names"]
    clin_mean     = np.array(ckpt["clin_mean"])
    clin_std      = np.array(ckpt["clin_std"])
    threshold     = float(ckpt.get("threshold", 0.5))
    # Complications come from checkpoint (authoritative) or fall back to config
    complications      = ckpt.get("complications") or list(cfg["model"].get("complications") or [])
    clinical_backbone  = ckpt.get("clinical_backbone") or cfg["model"].get("clinical_backbone", "mlp")
    tabm_cfg           = ckpt.get("tabm_cfg") or cfg["model"].get("tabm") or {}

    disable_cta = bool(cfg.get("training", {}).get("disable_cta", False))
    log(f"Threshold  : {threshold:.4f}  (from checkpoint)")
    log(f"Disable CTA: {disable_cta}")
    log(f"Clin feats : {len(feature_names)}")
    log(f"Complic.   : {complications or 'none'}\n")

    # ── data ──────────────────────────────────────────────────────────────────
    base_dir = Path(cfg["paths"]["base_dir"])
    size     = tuple(cfg["training"].get("image_size", [128, 128, 128]))
    spacing  = tuple(cfg["training"].get("voxel_spacing", [1.5, 1.5, 1.5]))

    splits, config, _ = load_splits_and_config(base_dir)
    test_ids          = splits[split]
    log(f"Cases      : {len(test_ids)}")

    test_files = build_data_list(test_ids, base_dir, config, feature_names, complications)
    cached_tfm = get_cached_transforms(size=size, spacing=spacing)
    test_tfm   = get_val_runtime_transforms(clin_mean, clin_std, complications)
    test_ds    = RuntimeDataset(
        CacheDataset(data=test_files, transform=cached_tfm,
                     cache_rate=1.0, num_workers=num_workers),
        test_tfm,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model = ReinterventionRiskModel(
        clinical_dim=len(feature_names),
        aaa_dim=cfg["model"].get("aaa_dim", 5),
        complications=complications,
        clinical_backbone=clinical_backbone,
        tabm_cfg=tabm_cfg,
    ).to(DEVICE)
    load_checkpoint(model, checkpoint_path, device=DEVICE)
    model.eval()
    log("Model loaded.\n")

    xai_engine = ReinterventionXAI(model, feature_names, complications)
    with_comp  = bool(complications)

    all_probs, all_gt     = [], []
    all_comp_probs: list  = []
    all_comp_gt: list     = []

    # ── per-patient loop ──────────────────────────────────────────────────────
    for i, batch in enumerate(test_loader):
        pid  = batch["pid"][0]
        cta  = batch["cta"].to(DEVICE).float()
        clin = batch["clin"].to(DEVICE).float()
        aaa  = batch["aaa_metrics"].to(DEVICE).float()
        gt   = int(batch["reintervention_label"].item())

        if disable_cta:
            cta = torch.zeros_like(cta)
        with torch.no_grad():
            outputs = model(cta, clin, aaa)
        prob = float(torch.sigmoid(outputs[0]).item())
        pred = int(prob > threshold)

        all_probs.append(prob)
        all_gt.append(gt)

        comp_probs_arr: np.ndarray | None = None
        if with_comp:
            comp_probs_arr = torch.sigmoid(outputs[1]).cpu().numpy().flatten()
            comp_gt_arr    = batch["complication_labels"].cpu().numpy().flatten()
            all_comp_probs.append(comp_probs_arr)
            all_comp_gt.append(comp_gt_arr)

        pred_info: dict = {
            "pid":                  pid,
            "reintervention_prob":  round(prob, 4),
            "reintervention_pred":  pred,
            "reintervention_gt":    gt,
            "label_type":           label_type(pred, gt),
            "threshold":            round(threshold, 4),
        }
        if with_comp and comp_probs_arr is not None:
            pred_info["complication_probs"] = {
                name: round(float(comp_probs_arr[j]), 4)
                for j, name in enumerate(complications)
            }
            comp_gt_arr = batch["complication_labels"].cpu().numpy().flatten()
            pred_info["complication_gt"] = {
                name: (int(comp_gt_arr[j]) if comp_gt_arr[j] != -1 else None)
                for j, name in enumerate(complications)
            }

        # Per-patient output directory
        pat_dir = out_dir / pid
        pat_dir.mkdir(parents=True, exist_ok=True)

        with open(pat_dir / "prediction.json", "w") as f:
            json.dump(pred_info, f, indent=2)

        # XAI
        xai_result = xai_engine.full_explain(cta, clin, aaa)
        with open(pat_dir / "xai.json", "w") as f:
            json.dump(xai_result, f, indent=2)

        with open(pat_dir / "xai_report.txt", "w") as f:
            f.write(format_xai_report(pid, pred_info, xai_result, complications))

        # GradCAM
        if gradcam:
            cam_np = xai_engine.gradcam(cta, clin, aaa)
            if cam_np is not None:
                xai_engine.save_gradcam_pngs(cta, cam_np, pat_dir)
            else:
                log(f"  [{pid}] GradCAM hook returned None.")

        # Console summary
        summary = (
            f"[{i+1:03d}/{len(test_ids)}] {pid:20s}  "
            f"prob={prob:.3f}  pred={pred}  gt={gt}  "
            f"{label_type(pred, gt):2s}  conf={xai_result['confidence_level']}"
        )
        if with_comp and "complication_probs" in pred_info:
            top = max(pred_info["complication_probs"], key=pred_info["complication_probs"].get)
            summary += f"  top={top}({pred_info['complication_probs'][top]:.2f})"
        log(summary)

    # ── overall metrics ───────────────────────────────────────────────────────
    metrics = compute_overall_metrics(
        all_probs, all_gt, threshold,
        all_comp_probs=all_comp_probs or None,
        all_comp_gt=all_comp_gt or None,
        complications=complications or None,
    )

    log("\n" + "=" * 62)
    log("TEST SET METRICS")
    log("=" * 62)
    for k, v in metrics.items():
        if k == "comp_auc_detail":
            log(f"  comp_auc_detail:")
            for comp, auc in v.items():
                log(f"    {comp:<24} {auc:.4f}")
        else:
            log(f"  {k:<25} {v}")

    save_metrics(metrics, out_dir)
    log(f"\nAll results written to: {out_dir}")
    log_file.close()
