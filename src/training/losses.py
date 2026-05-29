"""
Loss functions for the reintervention risk models.

ReinterventionLoss           — heteroscedastic focal BCE for the primary task.
MultiTaskReinterventionLoss  — same primary loss + masked per-complication loss.

Both use heteroscedastic NLL weighting with a KL-style symmetric prior:
    loss = exp(-logvar) * focal_bce + 0.5 * (exp(logvar) - 1 - logvar)
The prior has its minimum at logvar=0 and prevents calibration collapse
(logvar drifting to the clamp boundary, making train loss negative).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── shared helpers ────────────────────────────────────────────────────────────

def _focal_bce_per_sample(
    logit: torch.Tensor,
    gt: torch.Tensor,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample focal BCE, keeping the (B, 1) shape for heteroscedastic weighting."""
    smooth_gt = gt * (1 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(
        logit, smooth_gt, pos_weight=pos_weight, reduction="none"
    )
    p  = torch.sigmoid(logit).detach()
    pt = p * gt + (1 - p) * (1 - gt)
    return (1 - pt).pow(focal_gamma) * bce


def _heteroscedastic(
    logit: torch.Tensor,
    logvar: torch.Tensor,
    gt: torch.Tensor,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Heteroscedastic focal loss for one task. Returns a scalar.

    Uses a KL-style symmetric prior instead of the plain 0.5*logvar entropy term.
    0.5*(exp(logvar) - 1 - logvar) has its minimum at logvar=0 and penalises
    both logvar→+inf (exponential) and logvar→-inf (linear -logvar dominates).
    When train BCE→0 the optimal logvar→0, so train loss cannot go negative
    and the precision amplifier stays near 1× on memorised training samples.
    """
    bce       = _focal_bce_per_sample(logit, gt, focal_gamma, label_smoothing, pos_weight)
    logvar_c  = logvar.clamp(-4.0, 2.0)
    precision = torch.exp(-logvar_c)
    return (precision * bce).mean() + 0.5 * (logvar.exp() - 1 - logvar).mean()


# ── primary-task-only loss ────────────────────────────────────────────────────

class ReinterventionLoss(nn.Module):
    """
    For the no-complications model variant.
    Class imbalance is handled via pos_weight in the loss (no sampler in the dataloader).
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.focal_gamma     = focal_gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight_buf", torch.tensor([pos_weight], device=device))

    def forward(
        self,
        reint_logit: torch.Tensor,
        reint_logvar: torch.Tensor,
        reint_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        reint_gt_v = reint_gt.view_as(reint_logit)
        loss = _heteroscedastic(
            reint_logit, reint_logvar, reint_gt_v,
            self.focal_gamma, self.label_smoothing, self.pos_weight_buf,
        )
        return loss, {
            "loss_total":       loss.item(),
            "loss_reint":       loss.item(),
            "mean_uncertainty": reint_logvar.mean().item(),
        }


# ── multi-task loss ───────────────────────────────────────────────────────────

class MultiTaskReinterventionLoss(nn.Module):
    """
    For the with-complications model variant.

    Per-complication loss is masked for labels == -1 (missing ground truth).
    comp_weight controls the overall contribution of all complication losses.

    comp_loss_weights: per-complication scaling applied before summing
        (e.g. endoleak_2=1.0, endoleak_3=0.1).  Loaded from YAML.
    comp_pos_weights:  per-complication class-imbalance correction
        (N_neg/N_pos per complication in the training split).
    """

    def __init__(
        self,
        pos_weight_reint: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        comp_weight: float = 0.3,
        comp_loss_weights: list[float] | None = None,
        comp_pos_weights: list[float] | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.focal_gamma      = focal_gamma
        self.label_smoothing  = label_smoothing
        self.comp_weight      = comp_weight
        self.comp_loss_weights = comp_loss_weights  # plain list, not a buffer

        self.register_buffer("pos_weight_reint", torch.tensor([pos_weight_reint], device=device))
        if comp_pos_weights is not None:
            self.register_buffer(
                "comp_pos_weight_buf",
                torch.tensor(comp_pos_weights, dtype=torch.float32, device=device),
            )
        else:
            self.register_buffer("comp_pos_weight_buf", None)

    def forward(
        self,
        reint_logit: torch.Tensor,
        reint_logvar: torch.Tensor,
        comp_logits: torch.Tensor,
        comp_logvar: torch.Tensor,
        reint_gt: torch.Tensor,
        comp_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # Primary task
        loss_reint = _heteroscedastic(
            reint_logit, reint_logvar, reint_gt.view_as(reint_logit),
            self.focal_gamma, self.label_smoothing, self.pos_weight_reint,
        )

        # Per-complication, masked for unknown labels; weighted sum
        comp_losses: list[torch.Tensor] = []
        weights_used: list[float] = []
        for c in range(comp_logits.shape[1]):
            valid = comp_gt[:, c] != -1
            if valid.sum() == 0:
                continue
            pw = (
                self.comp_pos_weight_buf[c:c+1]
                if self.comp_pos_weight_buf is not None else None
            )
            loss_c = _heteroscedastic(
                comp_logits[valid, c].unsqueeze(1),
                comp_logvar[valid, c].unsqueeze(1),
                comp_gt[valid, c].unsqueeze(1),
                self.focal_gamma, self.label_smoothing,
                pos_weight=pw,
            )
            w = self.comp_loss_weights[c] if self.comp_loss_weights is not None else 1.0
            comp_losses.append(loss_c * w)
            weights_used.append(w)

        if comp_losses:
            loss_comp = torch.stack(comp_losses).sum() / max(sum(weights_used), 1e-8)
        else:
            loss_comp = torch.tensor(0.0, device=reint_logit.device)

        total = loss_reint + self.comp_weight * loss_comp

        return total, {
            "loss_total":       total.item(),
            "loss_reint":       loss_reint.item(),
            "loss_comp":        loss_comp.item(),
            "mean_uncertainty": reint_logvar.mean().item(),
        }
