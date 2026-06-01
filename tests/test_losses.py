"""
Tests for src/training/losses.py

Covers:
- _focal_bce_per_sample: shape, non-negativity, focal weighting
- ReinterventionLoss: forward output structure, device placement
- MultiTaskReinterventionLoss: forward, masking of -1 labels, comp_weight effect
"""

from __future__ import annotations

import pytest
import torch


# ── private helpers ───────────────────────────────────────────────────────────


def test_focal_bce_shape():
    from src.training.losses import _focal_bce_per_sample
    logit = torch.zeros(4, 1)
    gt    = torch.ones(4, 1)
    out   = _focal_bce_per_sample(logit, gt)
    assert out.shape == (4, 1)


def test_focal_bce_non_negative():
    from src.training.losses import _focal_bce_per_sample
    logit = torch.randn(8, 1)
    gt    = torch.randint(0, 2, (8, 1)).float()
    out   = _focal_bce_per_sample(logit, gt)
    assert (out >= 0).all()


def test_focal_bce_perfect_prediction_lower():
    """A perfectly confident correct prediction should have lower loss than a wrong one."""
    from src.training.losses import _focal_bce_per_sample
    # Predict 1 when gt=1 (large positive logit)
    correct = _focal_bce_per_sample(torch.tensor([[5.0]]), torch.tensor([[1.0]])).item()
    # Predict 0 when gt=1 (large negative logit)
    wrong   = _focal_bce_per_sample(torch.tensor([[-5.0]]), torch.tensor([[1.0]])).item()
    assert correct < wrong


def test_focal_bce_label_smoothing_effect():
    """Label smoothing should push loss away from 0 even for perfect predictions."""
    from src.training.losses import _focal_bce_per_sample
    smooth = _focal_bce_per_sample(
        torch.tensor([[10.0]]), torch.tensor([[1.0]]),
        label_smoothing=0.1
    ).item()
    hard = _focal_bce_per_sample(
        torch.tensor([[10.0]]), torch.tensor([[1.0]]),
        label_smoothing=0.0
    ).item()
    assert smooth > hard


# ── ReinterventionLoss ────────────────────────────────────────────────────────


def test_reintervention_loss_returns_tuple():
    from src.training.losses import ReinterventionLoss
    crit  = ReinterventionLoss(device="cpu")
    logit = torch.zeros(4, 1)
    gt    = torch.FloatTensor([0, 1, 0, 1]).unsqueeze(1)
    out   = crit(logit, gt)
    assert isinstance(out, tuple) and len(out) == 2


def test_reintervention_loss_scalar():
    from src.training.losses import ReinterventionLoss
    crit  = ReinterventionLoss(device="cpu")
    logit = torch.zeros(4, 1)
    gt    = torch.FloatTensor([0, 1, 0, 1]).unsqueeze(1)
    loss, _ = crit(logit, gt)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_reintervention_loss_dict_keys():
    from src.training.losses import ReinterventionLoss
    crit  = ReinterventionLoss(device="cpu")
    logit = torch.zeros(4, 1)
    gt    = torch.FloatTensor([0, 1, 0, 1]).unsqueeze(1)
    _, d  = crit(logit, gt)
    assert "loss_total" in d
    assert "loss_reint" in d
    assert "mean_uncertainty" not in d


def test_reintervention_loss_total_equals_reint():
    """Primary-task-only loss: total == reintervention component."""
    from src.training.losses import ReinterventionLoss
    crit  = ReinterventionLoss(device="cpu")
    logit = torch.zeros(4, 1)
    gt    = torch.FloatTensor([0, 1, 0, 1]).unsqueeze(1)
    _, d  = crit(logit, gt)
    assert d["loss_total"] == pytest.approx(d["loss_reint"], abs=1e-6)


def test_reintervention_loss_handles_gt_shape():
    """GT tensor with and without the trailing dim should both work."""
    from src.training.losses import ReinterventionLoss
    crit  = ReinterventionLoss(device="cpu")
    logit = torch.zeros(4, 1)
    gt_2d = torch.FloatTensor([0, 1, 0, 1]).unsqueeze(1)  # (4, 1)
    gt_1d = gt_2d.squeeze(1)                               # (4,)
    loss_2d, _ = crit(logit, gt_2d)
    loss_1d, _ = crit(logit, gt_1d)
    assert loss_2d.item() == pytest.approx(loss_1d.item(), rel=1e-4)


# ── MultiTaskReinterventionLoss ───────────────────────────────────────────────


def test_multitask_loss_returns_tuple():
    from src.training.losses import MultiTaskReinterventionLoss
    crit = MultiTaskReinterventionLoss(device="cpu")
    B, C = 4, 3
    out  = crit(
        torch.zeros(B, 1), torch.zeros(B, C),
        torch.ones(B, 1),  torch.zeros(B, C),
    )
    assert isinstance(out, tuple) and len(out) == 2


def test_multitask_loss_dict_keys():
    from src.training.losses import MultiTaskReinterventionLoss
    crit = MultiTaskReinterventionLoss(device="cpu")
    B, C = 4, 3
    _, d = crit(
        torch.zeros(B, 1), torch.zeros(B, C),
        torch.ones(B, 1),  torch.zeros(B, C),
    )
    for key in ("loss_total", "loss_reint", "loss_comp"):
        assert key in d
    assert "mean_uncertainty" not in d


def test_multitask_loss_all_comp_missing_zeroes_comp_loss():
    """When every complication label is -1, comp loss must be exactly 0."""
    from src.training.losses import MultiTaskReinterventionLoss
    crit    = MultiTaskReinterventionLoss(device="cpu")
    B, C    = 4, 3
    comp_gt = torch.full((B, C), -1.0)
    _, d    = crit(
        torch.zeros(B, 1), torch.zeros(B, C),
        torch.ones(B, 1),  comp_gt,
    )
    assert d["loss_comp"] == pytest.approx(0.0)


def test_multitask_loss_partial_masking():
    """Some -1 labels: comp loss is computed only on valid rows."""
    from src.training.losses import MultiTaskReinterventionLoss
    crit    = MultiTaskReinterventionLoss(device="cpu")
    B, C    = 4, 2
    comp_gt = torch.FloatTensor([
        [1.0, -1.0],
        [-1.0, 0.0],
        [0.0,  1.0],
        [-1.0, -1.0],
    ])
    _, d = crit(
        torch.zeros(B, 1), torch.zeros(B, C),
        torch.ones(B, 1),  comp_gt,
    )
    assert d["loss_comp"] > 0


def test_multitask_comp_weight_scales_total():
    """Doubling comp_weight should increase total loss when comp_loss > 0."""
    from src.training.losses import MultiTaskReinterventionLoss
    B, C         = 4, 2
    reint_logit  = torch.zeros(B, 1)
    comp_logits  = torch.ones(B, C)
    reint_gt     = torch.zeros(B, 1)
    comp_gt      = torch.ones(B, C)
    low_w,  _ = MultiTaskReinterventionLoss(comp_weight=0.05, device="cpu")(
        reint_logit, comp_logits, reint_gt, comp_gt)
    high_w, _ = MultiTaskReinterventionLoss(comp_weight=1.0,  device="cpu")(
        reint_logit, comp_logits, reint_gt, comp_gt)
    assert high_w.item() > low_w.item()


def test_multitask_loss_finite():
    from src.training.losses import MultiTaskReinterventionLoss
    crit    = MultiTaskReinterventionLoss(device="cpu")
    B, C    = 8, 3
    comp_gt = torch.randint(0, 2, (B, C)).float()
    loss, d = crit(
        torch.randn(B, 1), torch.randn(B, C),
        torch.randint(0, 2, (B, 1)).float(), comp_gt,
    )
    assert torch.isfinite(loss)
    for v in d.values():
        assert torch.isfinite(torch.tensor(v))
