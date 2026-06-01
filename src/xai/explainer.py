"""
ReinterventionXAI — unified explainability for both model variants.

Works for models trained with or without complications.
At test time, call save_gradcam_pngs() to write PNGs to disk.
During training, call get_gradcam_wandb_images() for W&B logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

AAA_METRIC_NAMES = [
    "volume_lumen_ml", "volume_ilt_ml", "volume_total_ml",
    "max_lumen_to_wall_dist_mm", "avg_lumen_to_wall_dist_mm",
]


class ReinterventionXAI:
    """
    Gradient-based and ablation-based explanations.

    Args:
        model:          Trained ReinterventionRiskModel (unwrapped, on correct device).
        feature_names:  Ordered list of clinical feature names (matches training order).
        complications:  List of complication names if model was trained with them.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: list[str],
        complications: list[str] | None = None,
    ) -> None:
        self.model         = model
        self.feature_names = np.array(feature_names)
        self.complications = list(complications or [])

    # ── public API ────────────────────────────────────────────────────────────

    def full_explain(
        self,
        cta: torch.Tensor,
        clin: torch.Tensor,
        aaa: torch.Tensor,
    ) -> dict:
        """
        Run all XAI components and return a single result dict.

        The dict is JSON-serialisable (numpy types converted to Python float/str).
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(cta, clin, aaa)

        reint_logit = outputs[0]
        reint_prob  = float(torch.sigmoid(reint_logit).item())
        # Confidence = distance from 0.5: 0 when maximally uncertain, 1 when certain.
        reint_conf  = 2.0 * abs(reint_prob - 0.5)

        result: dict = {
            "reintervention_prob":       round(reint_prob, 4),
            "reintervention_confidence": round(reint_conf, 4),
            "confidence_level":          self._conf_level(reint_conf),
        }

        if self.complications:
            comp_probs_t = torch.sigmoid(outputs[1]).squeeze().cpu().numpy()
            comp_prob_dict, comp_conf_dict = {}, {}
            for i, name in enumerate(self.complications):
                p = float(comp_probs_t[i] if comp_probs_t.ndim > 0 else comp_probs_t)
                c = 2.0 * abs(p - 0.5)
                comp_prob_dict[name] = round(p, 4)
                comp_conf_dict[name] = self._conf_level(c)
            result["complication_probs"]      = comp_prob_dict
            result["complication_confidence"] = comp_conf_dict

        clin_imp     = self._clinical_importance(cta, clin, aaa)
        aaa_imp      = self._aaa_importance(cta, clin, aaa)
        modality_imp = self._modality_importance(cta, clin, aaa)

        result.update({
            "clinical_importance": clin_imp,
            "aaa_importance":      aaa_imp,
            "modality_importance": modality_imp,
            "causal_chain":        self._causal_chain(result, clin_imp, modality_imp),
            "primary_risk_factor": self._primary_factor(clin_imp, result),
        })
        return result

    def gradcam(
        self,
        cta: torch.Tensor,
        clin: torch.Tensor,
        aaa: torch.Tensor,
    ) -> np.ndarray | None:
        """
        GradCAM on Merlin's last ResNet layer.
        Returns a (D, H, W) float32 array in [0, 1], or None if the hook fails.
        """
        i3res = self._get_i3res()
        if i3res is None:
            return None

        store: dict = {}

        def fwd_hook(module, inp, out):
            store["feat"] = out
            if out.requires_grad:
                out.retain_grad()

        # Patch checkpoint passthrough so gradients flow through all layers
        import torch.utils.checkpoint as cp_mod
        _orig = cp_mod.checkpoint
        cp_mod.checkpoint = lambda fn, *a, **kw: fn(*a)

        handle = i3res.layer4[-1].register_forward_hook(fwd_hook)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                out = self.model(cta=cta, clin=clin, aaa=aaa)
                out[0].sum().backward()  # grad w.r.t. reint_logit
        finally:
            handle.remove()
            cp_mod.checkpoint = _orig

        if "feat" not in store or store["feat"].grad is None:
            return None

        acts    = store["feat"].detach().cpu()
        grads   = store["feat"].grad.detach().cpu()
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam     = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam_up  = F.interpolate(cam, size=cta.shape[2:], mode="trilinear", align_corners=False)
        cam_np  = cam_up.squeeze().numpy()
        cam_np  = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np.astype(np.float32)

    def save_gradcam_pngs(
        self,
        cta: torch.Tensor,
        cam_np: np.ndarray,
        out_dir: Path,
    ) -> None:
        """Save axial / sagittal / coronal overlays as individual PNGs + one combined."""
        ct    = cta[0, 0].cpu().numpy()
        lumen = cta[0, 1].cpu().numpy() > 0.5
        ilt   = cta[0, 2].cpu().numpy() > 0.5
        D, H, W = ct.shape
        views = {
            "axial":    (ct[D//2],    cam_np[D//2],    lumen[D//2],    ilt[D//2]),
            "sagittal": (ct[:, H//2], cam_np[:, H//2], lumen[:, H//2], ilt[:, H//2]),
            "coronal":  (ct[:, :, W//2], cam_np[:, :, W//2],
                         lumen[:, :, W//2], ilt[:, :, W//2]),
        }

        # Combined 3-panel
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (vname, sl) in zip(axes, views.items()):
            self._render_slice(ax, *sl, f"GradCAM {vname}")
        fig.tight_layout(pad=0.5)
        fig.savefig(out_dir / "gradcam_combined.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Individual views
        for vname, sl in views.items():
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            self._render_slice(ax2, *sl, f"GradCAM {vname}")
            fig2.tight_layout(pad=0.1)
            fig2.savefig(out_dir / f"gradcam_{vname}.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)

    def get_gradcam_wandb_images(
        self,
        cta: torch.Tensor,
        cam_np: np.ndarray,
    ) -> dict:
        """Return {key: wandb.Image} for W&B logging during training."""
        import wandb

        ct    = cta[0, 0].cpu().numpy()
        lumen = cta[0, 1].cpu().numpy() > 0.5
        ilt   = cta[0, 2].cpu().numpy() > 0.5
        D, H, W = ct.shape
        views = {
            "axial":    (ct[D//2],    cam_np[D//2],    lumen[D//2],    ilt[D//2]),
            "sagittal": (ct[:, H//2], cam_np[:, H//2], lumen[:, H//2], ilt[:, H//2]),
            "coronal":  (ct[:, :, W//2], cam_np[:, :, W//2],
                         lumen[:, :, W//2], ilt[:, :, W//2]),
        }
        images = {}
        for vname, sl in views.items():
            fig, ax = plt.subplots(figsize=(4, 4))
            self._render_slice(ax, *sl, f"GradCAM {vname}")
            fig.tight_layout(pad=0.1)
            images[f"gradcam_{vname}"] = wandb.Image(fig)
            plt.close(fig)
        return images

    # ── importance methods ────────────────────────────────────────────────────

    def _clinical_importance(self, cta, clin, aaa) -> list[tuple]:
        clin_var = clin.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out   = self.model(cta, clin_var, aaa)
            score = torch.sigmoid(out[0]).mean()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        if clin_var.grad is None:
            return []
        grads = clin_var.grad.abs().squeeze().cpu().numpy()
        top   = np.argsort(grads)[::-1][:20]
        result = []
        for idx in top:
            s   = float(grads[idx])
            pct = float((grads >= s).mean())
            level = "HIGH" if pct > 0.9 else ("MED" if pct > 0.7 else "LOW")
            result.append((str(self.feature_names[idx]), round(s, 4), level))
        return result

    def _aaa_importance(self, cta, clin, aaa) -> list[tuple]:
        aaa_var = aaa.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out = self.model(cta, clin, aaa_var)
            self.model.zero_grad(set_to_none=True)
            torch.sigmoid(out[0]).mean().backward()
        if aaa_var.grad is None:
            return [(n, 0.0, "LOW") for n in AAA_METRIC_NAMES]
        grads = aaa_var.grad.abs().squeeze().cpu().numpy()
        grads = grads / (grads.max() + 1e-8)
        result = []
        for name, score in zip(AAA_METRIC_NAMES, grads.tolist()):
            level = "HIGH" if score > 0.7 else ("MED" if score > 0.3 else "LOW")
            result.append((name, round(score, 4), level))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def _modality_importance(self, cta, clin, aaa) -> dict[str, float]:
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

    # ── causal / narrative ────────────────────────────────────────────────────

    def _causal_chain(self, result: dict, clin_imp: list, modality_imp: dict) -> str:
        reint_prob = result["reintervention_prob"]
        top_clin   = clin_imp[0][0] if clin_imp else "clinical features"
        top_mod    = max(modality_imp, key=modality_imp.get) if modality_imp else "multimodal data"

        if self.complications and "complication_probs" in result:
            comp_probs = result["complication_probs"]
            comp_conf  = result.get("complication_confidence", {})
            top_comps  = [
                (n, p) for n, p in
                sorted(comp_probs.items(), key=lambda x: x[1], reverse=True)
                if p > 0.15
            ][:3]
            comp_str = ", ".join(
                f"{n} ({p:.0%}, {comp_conf.get(n, '?')} conf)" for n, p in top_comps
            ) or "no dominant complication"
            if reint_prob > 0.5:
                return (f"Elevated {top_clin} via {top_mod} → "
                        f"probable complications: {comp_str} → "
                        f"likely reintervention ({reint_prob:.0%})")
            return (f"{top_clin} and {comp_str} — monitoring advised "
                    f"(reintervention risk {reint_prob:.0%})")

        if reint_prob > 0.5:
            return (f"Elevated {top_clin} detected by {top_mod} → "
                    f"likely reintervention needed ({reint_prob:.0%})")
        return (f"{top_clin} via {top_mod} — monitoring advised, "
                f"low reintervention risk ({reint_prob:.0%})")

    def _primary_factor(self, clin_imp: list, result: dict) -> str:
        if self.complications and "complication_probs" in result:
            comp_probs = result["complication_probs"]
            top_comp   = max(comp_probs, key=comp_probs.get) if comp_probs else None
            if top_comp and comp_probs[top_comp] > 0.4:
                return f"Probable {top_comp} ({comp_probs[top_comp]:.0%})"
        if clin_imp:
            return f"{clin_imp[0][0]} ({clin_imp[0][2]} impact)"
        return "Unknown"

    # ── internal helpers ──────────────────────────────────────────────────────

    def _conf_level(self, conf: float) -> str:
        """conf = 2*|prob - 0.5|, ranges 0 (uncertain) to 1 (certain)."""
        if conf > 0.6: return "HIGH"
        if conf > 0.3: return "MED"
        return "LOW"

    def _get_i3res(self) -> nn.Module | None:
        """Find the I3ResNet inside the LoRA/PEFT wrapper."""
        for _, mod in self.model.visual_branch.named_modules():
            if hasattr(mod, "layer4") and hasattr(mod, "layer1"):
                return mod
        return None

    @staticmethod
    def _render_slice(
        ax,
        ct_sl: np.ndarray,
        cam_sl: np.ndarray,
        lumen_sl: np.ndarray,
        ilt_sl: np.ndarray,
        title: str,
    ) -> None:
        ax.imshow(ct_sl,  cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(cam_sl, cmap="jet",  alpha=0.40, vmin=0, vmax=1, interpolation="nearest")
        lr = np.zeros((*lumen_sl.shape, 4)); lr[lumen_sl] = [0.0, 1.0, 1.0, 0.35]
        ir = np.zeros((*ilt_sl.shape,   4)); ir[ilt_sl]   = [1.0, 1.0, 0.0, 0.35]
        ax.imshow(lr, interpolation="nearest")
        ax.imshow(ir, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
