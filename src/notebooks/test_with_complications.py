#!/usr/bin/env python3
"""
Test script for AAA Reintervention Risk Model (with-complications variant).

Loads a trained checkpoint, runs predictions on every patient in the chosen split,
and writes per-patient XAI outputs + overall metrics including per-complication AUC.

Output layout
─────────────
<output_dir>/
  <pid>/
    prediction.json       # reint prob/pred/GT + per-complication probs
    xai.json              # full XAI: clinical/AAA/modality importance + causal chain
    xai_report.txt        # human-readable report
    gradcam_axial.png     # only with --gradcam
    gradcam_sagittal.png
    gradcam_coronal.png
    gradcam_combined.png
  test_metrics.json       # overall test-set metrics + per-complication AUC
  test_metrics.csv        # flat CSV for Excel
  test_log.txt            # full console log

Usage
─────
python test_with_complications.py \
    --checkpoint /home/ranema/AI4EVAR/checkpoints_reint_comp/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/with_complications \
    [--gradcam] [--split test] [--workers 8]
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, Resized, MapTransform, ToTensord,
)
from peft import LoraConfig, get_peft_model
from merlin import Merlin
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, precision_recall_curve, confusion_matrix,
)

# ── constants ──────────────────────────────────────────────────────────────

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1")
SIZE     = (128, 128, 128)

COMPLICATIONS = [
    "endoleak_1a", "endoleak_1b", "endoleak_2", "endoleak_3", "endoleak_5",
    "migration", "infection",
]
NUM_COMPLICATIONS = len(COMPLICATIONS)

AAA_METRIC_NAMES = [
    "volume_lumen_ml", "volume_ilt_ml", "volume_total_ml",
    "max_lumen_to_wall_dist_mm", "avg_lumen_to_wall_dist_mm",
]

# ── data ───────────────────────────────────────────────────────────────────

def clean_clinical_features(feat_dict):
    cleaned = {}
    for k, v in feat_dict.items():
        try:
            cleaned[k] = float(v) if v not in ["", None] else 0.0
        except Exception:
            cleaned[k] = 0.0
    return cleaned


def build_data_list(ids, base_dir, config, feature_keys):
    data = []
    for pid in ids:
        pdata     = config.get(pid, {})
        clin_dict = pdata.get("pre_op_features", {})
        clin = np.nan_to_num(
            np.array([clin_dict.get(k, 0.0) for k in feature_keys], dtype=np.float32)
        )
        metrics_path = base_dir / "metricsTr" / f"{pid}.json"
        aaa_metrics  = [0.0] * 5
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
                aaa_metrics = [
                    m.get("volume_lumen_ml", 0.0),
                    m.get("volume_ilt_ml", 0.0),
                    m.get("volume_total_ml", 0.0),
                    m.get("max_lumen_to_wall_dist_mm", 0.0),
                    m.get("avg_lumen_to_wall_dist_mm", 0.0),
                ]
        gt          = pdata.get("ground_truth", {})
        reint_label = float(gt.get("reintervention", 0))
        comp_labels = np.array(
            [float(gt.get(comp, -1.0)) for comp in COMPLICATIONS], dtype=np.float32
        )
        data.append({
            "ct":                    str(base_dir / "imagesTr"   / f"{pid}.nii.gz"),
            "lbl":                   str(base_dir / "labelsTr"   / f"{pid}.nii.gz"),
            "dst":                   str(base_dir / "distMapsTr" / f"{pid}.nii.gz"),
            "clin":                  clin,
            "aaa_metrics":           aaa_metrics,
            "reintervention_label":  np.array([reint_label], dtype=np.float32),
            "complication_labels":   comp_labels,
            "pid":                   pid,
        })
    return data


class BuildCTAd(MapTransform):
    def __call__(self, data):
        ct, lbl, dst = data["ct"], data["lbl"], data["dst"]
        data["cta"] = torch.cat([ct, (lbl == 1).float(), (lbl == 2).float(), dst], dim=0)
        return data


class NormalizeClinicald(MapTransform):
    def __init__(self, keys, mean, std, eps=1e-8):
        super().__init__(keys)
        self.mean = torch.tensor(mean)
        self.std  = torch.tensor(std)
        self.eps  = eps

    def __call__(self, data):
        clin = torch.as_tensor(data["clin"]).float()
        data["clin"] = (clin - self.mean) / (self.std + self.eps)
        return data


def get_cached_transforms():
    return Compose([
        LoadImaged(keys=["ct", "lbl", "dst"]),
        EnsureChannelFirstd(keys=["ct", "lbl", "dst"]),
        Orientationd(keys=["ct", "lbl", "dst"], axcodes="RAS"),
        Spacingd(
            keys=["ct", "lbl", "dst"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest", "bilinear"),
        ),
        ScaleIntensityRanged(keys=["ct"], a_min=-200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        Resized(
            keys=["ct", "lbl", "dst"],
            spatial_size=SIZE,
            mode=("trilinear", "nearest", "trilinear"),
        ),
    ])


def get_test_runtime_transforms(clin_mean, clin_std):
    return Compose([
        BuildCTAd(keys=["ct", "lbl", "dst"]),
        ToTensord(keys=["ct", "lbl", "dst", "aaa_metrics", "reintervention_label", "complication_labels"]),
        NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
    ])


class RuntimeDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, runtime_transform):
        self.base_ds = base_ds
        self.runtime = runtime_transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        return self.runtime(self.base_ds[idx])


# ── model (identical architecture to training script) ──────────────────────

def disable_checkpointing(model):
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False
        if hasattr(m, "enable_gradient_checkpointing"):
            try:
                m.enable_gradient_checkpointing(False)
            except Exception:
                pass


class ReinterventionRiskModel(nn.Module):
    def __init__(self, clinical_dim=744, aaa_dim=5, num_complications=NUM_COMPLICATIONS):
        super().__init__()
        self.num_complications = num_complications
        self.cta_adapter = nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1), nn.ReLU(), nn.Conv3d(8, 1, 1)
        )
        self.visual_base   = Merlin(ImageEmbedding=True)
        disable_checkpointing(self.visual_base)
        self.visual_branch = get_peft_model(
            self.visual_base,
            LoraConfig(r=8, target_modules=["query", "value", "query_global", "value_global"]),
        )
        self.visual_proj   = nn.Linear(2048, 256)
        self.feature_gate  = nn.Parameter(torch.zeros(clinical_dim))
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.Dropout(0.2),
        )
        self.aaa_metrics_branch = nn.Sequential(
            nn.Linear(aaa_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        total_fusion_dim = 256 + 32 + 32
        self.condition_net = nn.Sequential(
            nn.Linear(total_fusion_dim, 512), nn.ReLU(),
            nn.Linear(512, 2 * total_fusion_dim),
        )
        self.fusion_norm = nn.LayerNorm(total_fusion_dim)
        self.reintervention_head = nn.Sequential(
            nn.Linear(total_fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
        self.complications_head = nn.Sequential(
            nn.Linear(total_fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_complications * 2),
        )

    def film_modulate(self, x, gamma_beta):
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return x * (1 + torch.tanh(gamma) * 0.1) + torch.tanh(beta) * 0.1

    def encode_visual(self, cta):
        v = self.visual_branch(cta)
        if isinstance(v, (tuple, list)):
            v = v[0]
        if v.dim() == 3 and v.shape[0] == 1:
            v = v.squeeze(0)
        return v.flatten(1) if v.dim() > 2 else v

    def forward(self, cta=None, clin=None, aaa=None, return_fused=False):
        cta   = self.cta_adapter(cta)
        v_emb = self.visual_proj(self.encode_visual(cta))
        c_emb = self.clinical_branch(clin * torch.sigmoid(self.feature_gate))
        a_emb = self.aaa_metrics_branch(aaa)
        fused = torch.cat([v_emb, c_emb, a_emb], dim=1)
        cond  = self.condition_net(fused)
        fused = self.film_modulate(fused, cond)
        fused = self.fusion_norm(fused)
        if return_fused:
            return fused
        reint_out    = self.reintervention_head(fused)
        reint_logit  = reint_out[:, 0:1]
        reint_logvar = torch.clamp(reint_out[:, 1:2], -2, 2)
        comp_out     = self.complications_head(fused)
        comp_out     = comp_out.view(comp_out.size(0), self.num_complications, 2)
        comp_logits  = comp_out[:, :, 0]
        comp_logvar  = torch.clamp(comp_out[:, :, 1], -2, 2)
        return reint_logit, reint_logvar, comp_logits, comp_logvar


# ── XAI ────────────────────────────────────────────────────────────────────

class TestXAI:
    def __init__(self, model, feature_names):
        self.model         = model
        self.feature_names = np.array(feature_names)

    def full_explain(self, cta, clin, aaa) -> dict:
        self.model.eval()

        with torch.no_grad():
            reint_logit, reint_logvar, comp_logits, comp_logvar = self.model(cta, clin, aaa)

        reint_prob = float(torch.sigmoid(reint_logit).item())
        reint_std  = float(torch.exp(0.5 * reint_logvar).item())
        reint_conf = 1.0 / (1.0 + reint_std)

        comp_probs_np  = torch.sigmoid(comp_logits).squeeze().cpu().numpy()
        comp_logvar_np = comp_logvar.squeeze().cpu().numpy()
        comp_prob_dict = {}
        comp_conf_dict = {}
        for i, name in enumerate(COMPLICATIONS):
            p  = float(comp_probs_np[i] if comp_probs_np.ndim > 0 else comp_probs_np)
            lv = float(comp_logvar_np[i] if comp_logvar_np.ndim > 0 else comp_logvar_np)
            c  = 1.0 / (1.0 + float(np.exp(0.5 * lv)))
            comp_prob_dict[name] = round(p, 4)
            comp_conf_dict[name] = self._conf_level(c)

        clin_imp     = self._clinical_importance(cta, clin, aaa)
        aaa_imp      = self._aaa_importance(cta, clin, aaa)
        modality_imp = self._modality_importance(cta, clin, aaa)
        causal_chain = self._causal_chain(reint_prob, comp_prob_dict, comp_conf_dict, clin_imp, modality_imp)
        primary      = self._primary_factor(clin_imp, comp_prob_dict)

        return {
            "reintervention_prob":       reint_prob,
            "reintervention_confidence": round(reint_conf, 4),
            "confidence_level":          self._conf_level(reint_conf),
            "complication_probs":        comp_prob_dict,
            "complication_confidence":   comp_conf_dict,
            "clinical_importance":       clin_imp,
            "aaa_importance":            aaa_imp,
            "modality_importance":       modality_imp,
            "causal_chain":              causal_chain,
            "primary_risk_factor":       primary,
        }

    def _conf_level(self, conf: float) -> str:
        if conf > 0.75: return "HIGH"
        if conf > 0.55: return "MED"
        return "LOW"

    def _clinical_importance(self, cta, clin, aaa) -> List[Tuple]:
        clin_var = clin.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            reint_logit, *_ = self.model(cta, clin_var, aaa)
            score = torch.sigmoid(reint_logit).mean()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        if clin_var.grad is None:
            return []
        grads = clin_var.grad.abs().squeeze().cpu().numpy()
        top   = np.argsort(grads)[::-1][:20]
        result = []
        for idx in top:
            s     = float(grads[idx])
            pct   = float((grads >= s).mean())
            level = "HIGH" if pct > 0.9 else ("MED" if pct > 0.7 else "LOW")
            result.append((self.feature_names[idx], round(s, 4), level))
        return result

    def _aaa_importance(self, cta, clin, aaa) -> List[Tuple]:
        aaa_var = aaa.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            reint_logit, *_ = self.model(cta, clin, aaa_var)
            self.model.zero_grad(set_to_none=True)
            torch.sigmoid(reint_logit).mean().backward()
        if aaa_var.grad is None:
            return [(name, 0.0, "LOW") for name in AAA_METRIC_NAMES]
        grads = aaa_var.grad.abs().squeeze().cpu().numpy()
        grads = grads / (grads.max() + 1e-8)
        result = []
        for name, score in zip(AAA_METRIC_NAMES, grads.tolist()):
            level = "HIGH" if score > 0.7 else ("MED" if score > 0.3 else "LOW")
            result.append((name, round(score, 4), level))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def _modality_importance(self, cta, clin, aaa) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            base_p  = float(torch.sigmoid(self.model(cta, clin, aaa)[0]))
            no_cta  = float(torch.sigmoid(self.model(torch.zeros_like(cta), clin, aaa)[0]))
            no_clin = float(torch.sigmoid(self.model(cta, torch.zeros_like(clin), aaa)[0]))
            no_aaa  = float(torch.sigmoid(self.model(cta, clin, torch.zeros_like(aaa))[0]))
        impacts = {
            "CTA":         abs(base_p - no_cta),
            "Clinical":    abs(base_p - no_clin),
            "AAA_metrics": abs(base_p - no_aaa),
        }
        total = sum(impacts.values()) or 1.0
        return {k: round(v / total, 4) for k, v in impacts.items()}

    def _causal_chain(self, reint_prob, comp_probs, comp_conf, clin_imp, modality_imp) -> str:
        top_comps = [
            (name, prob) for name, prob in
            sorted(comp_probs.items(), key=lambda x: x[1], reverse=True)
            if prob > 0.15
        ][:3]
        top_clin = clin_imp[0][0] if clin_imp else "clinical features"
        top_mod  = max(modality_imp, key=modality_imp.get) if modality_imp else "multimodal data"
        comp_str = ", ".join(
            f"{n} ({p:.0%}, {comp_conf.get(n, '?')} conf)" for n, p in top_comps
        ) or "no dominant complication"
        if reint_prob > 0.5:
            return (f"Elevated {top_clin} detected via {top_mod} → "
                    f"probable complications: {comp_str} → "
                    f"likely reintervention ({reint_prob:.0%})")
        return (f"{top_clin} and {comp_str} — watchful monitoring advised "
                f"(reintervention risk {reint_prob:.0%})")

    def _primary_factor(self, clin_imp, comp_probs) -> str:
        top_comp = max(comp_probs, key=comp_probs.get) if comp_probs else None
        if top_comp and comp_probs[top_comp] > 0.4:
            return f"Probable {top_comp} ({comp_probs[top_comp]:.0%})"
        if clin_imp:
            return f"{clin_imp[0][0]} ({clin_imp[0][2]} impact)"
        return "Unknown"

    # ── GradCAM ────────────────────────────────────────────────────────────

    def _get_i3res(self):
        for _, mod in self.model.visual_branch.named_modules():
            if hasattr(mod, "layer4") and hasattr(mod, "layer1"):
                return mod
        return None

    def gradcam(self, cta, clin, aaa):
        """Returns (D,H,W) numpy heatmap in [0,1] or None on failure."""
        i3res = self._get_i3res()
        if i3res is None:
            return None
        store = {}

        def fwd_hook(module, inp, out):
            store["feat"] = out
            if out.requires_grad:
                out.retain_grad()

        import torch.utils.checkpoint as cp_module
        _orig = cp_module.checkpoint
        cp_module.checkpoint = lambda fn, *a, **kw: fn(*a)

        handle = i3res.layer4[-1].register_forward_hook(fwd_hook)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                reint_logit, *_ = self.model(cta=cta, clin=clin, aaa=aaa)
                reint_logit.sum().backward()
        finally:
            handle.remove()
            cp_module.checkpoint = _orig

        if "feat" not in store or store["feat"].grad is None:
            return None

        acts    = store["feat"].detach().cpu()
        grads   = store["feat"].grad.detach().cpu()
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam     = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam_up  = F.interpolate(cam, size=cta.shape[2:], mode="trilinear", align_corners=False)
        cam_np  = cam_up.squeeze().numpy()
        cam_np  = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np

    def save_gradcam_pngs(self, cta, cam_np, out_dir: Path):
        ct    = cta[0, 0].cpu().numpy()
        lumen = cta[0, 1].cpu().numpy() > 0.5
        ilt   = cta[0, 2].cpu().numpy() > 0.5
        D, H, W = ct.shape
        views = {
            "axial":    (ct[D//2],    cam_np[D//2],    lumen[D//2],    ilt[D//2]),
            "sagittal": (ct[:, H//2], cam_np[:, H//2], lumen[:, H//2], ilt[:, H//2]),
            "coronal":  (ct[:, :, W//2], cam_np[:, :, W//2], lumen[:, :, W//2], ilt[:, :, W//2]),
        }

        def _render(ax, ct_sl, cam_sl, lumen_sl, ilt_sl, title):
            ax.imshow(ct_sl,  cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.imshow(cam_sl, cmap="jet",  alpha=0.40, vmin=0, vmax=1, interpolation="nearest")
            lr = np.zeros((*lumen_sl.shape, 4)); lr[lumen_sl] = [0.0, 1.0, 1.0, 0.35]
            ir = np.zeros((*ilt_sl.shape,   4)); ir[ilt_sl]   = [1.0, 1.0, 0.0, 0.35]
            ax.imshow(lr, interpolation="nearest")
            ax.imshow(ir, interpolation="nearest")
            ax.set_title(title, fontsize=9); ax.axis("off")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (vname, sl) in zip(axes, views.items()):
            _render(ax, *sl, f"GradCAM {vname}")
        fig.tight_layout(pad=0.5)
        fig.savefig(out_dir / "gradcam_combined.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        for vname, sl in views.items():
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            _render(ax2, *sl, f"GradCAM {vname}")
            fig2.tight_layout(pad=0.1)
            fig2.savefig(out_dir / f"gradcam_{vname}.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)


# ── reporting ──────────────────────────────────────────────────────────────

def _label_type(pred, gt) -> str:
    if pred == 1 and gt == 1: return "TP"
    if pred == 1 and gt == 0: return "FP"
    if pred == 0 and gt == 0: return "TN"
    return "FN"


def format_xai_report(pid, pred_info, xai) -> str:
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
        "",
        "COMPLICATION BREAKDOWN (sorted by probability):",
    ]
    for comp, prob in sorted(xai["complication_probs"].items(), key=lambda x: x[1], reverse=True):
        conf = xai["complication_confidence"].get(comp, "?")
        lines.append(f"  • {comp:<20}  {prob:.1%}  [{conf} confidence]")
    lines += ["", "TOP CLINICAL FACTORS (gradient-based, top 10):"]
    for feat, score, level in xai["clinical_importance"][:10]:
        lines.append(f"  • {feat:<42} {level:4} ({score:.4f})")
    lines += ["", "AAA MORPHOLOGY IMPORTANCE:"]
    for metric, score, level in xai["aaa_importance"]:
        lines.append(f"  • {metric:<42} {level:4} ({score:.4f})")
    lines += ["", "MODALITY IMPORTANCE (ablation):"]
    for mod, imp in sorted(xai["modality_importance"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  • {mod:<20} {imp:.1%}")
    lines += [
        "",
        "CAUSAL CHAIN:",
        f"  {xai['causal_chain']}",
        "",
        "PRIMARY RISK FACTOR:",
        f"  {xai['primary_risk_factor']}",
        "",
    ]
    return "\n".join(lines)


def compute_overall_metrics(all_probs, all_gt, all_comp_probs, all_comp_gt, threshold) -> dict:
    probs = np.array(all_probs)
    gt    = np.array(all_gt).astype(int)
    preds = (probs > threshold).astype(int)

    p_c, r_c, t_c = precision_recall_curve(gt, probs)
    f1s           = 2 * p_c[:-1] * r_c[:-1] / (p_c[:-1] + r_c[:-1] + 1e-8)
    best_f1_idx   = int(np.argmax(f1s))

    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    metrics = {
        "n_samples":       len(gt),
        "n_positive":      int(gt.sum()),
        "n_negative":      int((~gt.astype(bool)).sum()),
        "threshold_used":  round(float(threshold), 4),
        "auc":             round(float(roc_auc_score(gt, probs)), 4) if len(np.unique(gt)) > 1 else 0.5,
        "f1":              round(float(f1_score(gt, preds)), 4),
        "precision":       round(float(precision_score(gt, preds, zero_division=0)), 4),
        "recall":          round(float(recall_score(gt, preds, zero_division=0)), 4),
        "accuracy":        round(float(accuracy_score(gt, preds)), 4),
        "best_f1":         round(float(f1s[best_f1_idx]), 4),
        "best_threshold":  round(float(t_c[best_f1_idx]), 4),
        "TP":              int(tp),
        "FP":              int(fp),
        "TN":              int(tn),
        "FN":              int(fn),
        "TPR_sensitivity": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
        "TNR_specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        "FPR_fallout":     round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
        "FNR_missrate":    round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4),
    }

    # Per-complication AUC (only where both classes present and GT is known)
    if all_comp_probs:
        comp_p = np.stack(all_comp_probs)  # (N, C)
        comp_g = np.stack(all_comp_gt)     # (N, C)
        comp_aucs = {}
        for i, name in enumerate(COMPLICATIONS):
            valid = comp_g[:, i] != -1
            if valid.sum() < 2:
                continue
            y_t = comp_g[valid, i]
            y_p = comp_p[valid, i]
            if len(np.unique(y_t)) > 1:
                comp_aucs[name] = round(float(roc_auc_score(y_t, y_p)), 4)
        if comp_aucs:
            metrics["comp_mean_auc"]  = round(float(np.mean(list(comp_aucs.values()))), 4)
            metrics["comp_auc_detail"] = comp_aucs

    return metrics


# ── main ───────────────────────────────────────────────────────────────────

def run_test(args):
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "test_log.txt", "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Checkpoint : {args.checkpoint}")
    log(f"Output dir : {out_dir}")
    log(f"Split      : {args.split}")
    log(f"GradCAM    : {args.gradcam}")
    log(f"Device     : {DEVICE}")

    ckpt          = torch.load(args.checkpoint, map_location=DEVICE)
    feature_names = ckpt["feature_names"]
    clin_mean     = np.array(ckpt["clin_mean"])
    clin_std      = np.array(ckpt["clin_std"])
    threshold     = float(ckpt.get("threshold", 0.5))
    log(f"Threshold  : {threshold:.4f}  (from checkpoint)")
    log(f"Clin feats : {len(feature_names)}")

    with open(BASE_DIR / "dataset_splits.json") as f:
        splits = json.load(f)
    with open(BASE_DIR / "clinical_data/radar_multimodal_config_v1.json") as f:
        config = json.load(f)

    for pid in config:
        if "pre_op_features" in config[pid]:
            config[pid]["pre_op_features"] = clean_clinical_features(config[pid]["pre_op_features"])

    test_ids = splits[args.split]
    log(f"Cases      : {len(test_ids)}\n")

    test_files = build_data_list(test_ids, BASE_DIR, config, feature_names)
    _cached    = get_cached_transforms()
    test_ds    = RuntimeDataset(
        CacheDataset(data=test_files, transform=_cached, cache_rate=1.0, num_workers=args.workers),
        get_test_runtime_transforms(clin_mean, clin_std),
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = ReinterventionRiskModel(clinical_dim=len(feature_names)).to(DEVICE)
    state_dict = ckpt["state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    log("Model loaded.\n")

    xai_engine = TestXAI(model, feature_names)

    all_probs, all_gt     = [], []
    all_comp_probs, all_comp_gt = [], []

    for i, batch in enumerate(test_loader):
        pid      = batch["pid"][0]
        cta      = batch["cta"].to(DEVICE).float()
        clin     = batch["clin"].to(DEVICE).float()
        aaa      = batch["aaa_metrics"].to(DEVICE).float()
        gt       = int(batch["reintervention_label"].item())
        comp_gt  = batch["complication_labels"].cpu().numpy().flatten()

        with torch.no_grad():
            reint_logit, reint_logvar, comp_logits, comp_logvar = model(cta, clin, aaa)
        prob       = float(torch.sigmoid(reint_logit).item())
        pred       = int(prob > threshold)
        comp_probs = torch.sigmoid(comp_logits).cpu().numpy().flatten()

        all_probs.append(prob)
        all_gt.append(gt)
        all_comp_probs.append(comp_probs)
        all_comp_gt.append(comp_gt)

        pred_info = {
            "pid":                  pid,
            "reintervention_prob":  round(prob, 4),
            "reintervention_pred":  pred,
            "reintervention_gt":    gt,
            "label_type":           _label_type(pred, gt),
            "threshold":            round(threshold, 4),
            "reint_logvar":         round(float(reint_logvar.item()), 4),
            "complication_probs":   {
                name: round(float(comp_probs[j]), 4)
                for j, name in enumerate(COMPLICATIONS)
            },
            "complication_gt":      {
                name: int(comp_gt[j]) if comp_gt[j] != -1 else None
                for j, name in enumerate(COMPLICATIONS)
            },
        }

        pat_dir = out_dir / pid
        pat_dir.mkdir(parents=True, exist_ok=True)

        with open(pat_dir / "prediction.json", "w") as f:
            json.dump(pred_info, f, indent=2)

        # Full XAI
        xai_result = xai_engine.full_explain(cta, clin, aaa)

        if args.gradcam:
            cam_np = xai_engine.gradcam(cta, clin, aaa)
            if cam_np is not None:
                xai_engine.save_gradcam_pngs(cta, cam_np, pat_dir)

        with open(pat_dir / "xai.json", "w") as f:
            json.dump(xai_result, f, indent=2)

        report = format_xai_report(pid, pred_info, xai_result)
        with open(pat_dir / "xai_report.txt", "w") as f:
            f.write(report)

        top_comp = max(pred_info["complication_probs"], key=pred_info["complication_probs"].get)
        log(
            f"[{i+1:03d}/{len(test_ids)}] {pid:20s}  "
            f"prob={prob:.3f}  pred={pred}  gt={gt}  "
            f"{_label_type(pred, gt):2s}  conf={xai_result['confidence_level']}  "
            f"top_comp={top_comp}({pred_info['complication_probs'][top_comp]:.2f})"
        )

    # Overall metrics
    metrics = compute_overall_metrics(all_probs, all_gt, all_comp_probs, all_comp_gt, threshold)

    log("\n" + "=" * 60)
    log("TEST SET METRICS")
    log("=" * 60)
    for k, v in metrics.items():
        if k == "comp_auc_detail":
            log(f"  {'comp_auc_detail':<25}")
            for comp, auc in v.items():
                log(f"    {comp:<22} {auc:.4f}")
        else:
            log(f"  {k:<25} {v}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Flat CSV (comp_auc_detail is flattened)
    with open(out_dir / "test_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            if k == "comp_auc_detail":
                for comp, auc in v.items():
                    w.writerow([f"auc_{comp}", auc])
            else:
                w.writerow([k, v])

    log(f"\nAll results written to: {out_dir}")
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AAA reintervention model — with complications")
    parser.add_argument("--checkpoint",
        default="/home/ranema/AI4EVAR/checkpoints_reint_comp/best_stage_b.pt")
    parser.add_argument("--output_dir",
        default="/home/ranema/AI4EVAR/test_results/with_complications")
    parser.add_argument("--gradcam", action="store_true",
        help="Enable GradCAM (adds ~10-30 s per patient, requires backward pass)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_test(args)
