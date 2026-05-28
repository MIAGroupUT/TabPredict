"""
Tests for src/models/reintervention.py

Covers:
- Module constants
- Model forward-pass output shapes (no-comp and with-comp variants)
- return_fused flag
- FiLM modulation effect
- get_model, disable_checkpointing helpers
- load_checkpoint: normal load, DataParallel prefix stripping, legacy key filtering
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from tests.conftest import BATCH, CLINICAL_DIM, AAA_DIM, SPATIAL, COMPLICATIONS, N_COMP


# ── constants ─────────────────────────────────────────────────────────────────


def test_complications_full_length():
    from src.models.reintervention import COMPLICATIONS_FULL
    assert len(COMPLICATIONS_FULL) == 7


def test_complications_no_comp_empty():
    from src.models.reintervention import COMPLICATIONS_NO_COMP
    assert COMPLICATIONS_NO_COMP == []


def test_aaa_metric_names_length():
    from src.models.reintervention import AAA_METRIC_NAMES
    assert len(AAA_METRIC_NAMES) == 5


def test_aaa_metric_names_content():
    from src.models.reintervention import AAA_METRIC_NAMES
    assert "volume_lumen_ml" in AAA_METRIC_NAMES
    assert "max_lumen_to_wall_dist_mm" in AAA_METRIC_NAMES


# ── helpers ───────────────────────────────────────────────────────────────────


def test_get_model_unwraps_dataparallel(model_no_comp):
    from src.models.reintervention import get_model
    wrapped = nn.DataParallel(model_no_comp)
    assert get_model(wrapped) is model_no_comp


def test_get_model_passthrough(model_no_comp):
    from src.models.reintervention import get_model
    assert get_model(model_no_comp) is model_no_comp


def test_disable_checkpointing_runs(model_no_comp):
    from src.models.reintervention import disable_checkpointing
    # Should complete without error regardless of which attrs modules have
    disable_checkpointing(model_no_comp)


def test_disable_checkpointing_sets_flag():
    """Modules with gradient_checkpointing get it set to False."""
    from src.models.reintervention import disable_checkpointing

    class GradCkptModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = True

    m = GradCkptModule()
    disable_checkpointing(m)
    assert m.gradient_checkpointing is False


# ── forward — no complications variant ───────────────────────────────────────


def test_forward_no_comp_output_count(model_no_comp, cta, clin, aaa):
    out = model_no_comp(cta, clin, aaa)
    assert len(out) == 2, "no-comp model must return exactly 2 tensors"


def test_forward_no_comp_shapes(model_no_comp, cta, clin, aaa):
    logit, logvar = model_no_comp(cta, clin, aaa)
    assert logit.shape  == (BATCH, 1)
    assert logvar.shape == (BATCH, 1)


def test_forward_no_comp_logvar_clamped(model_no_comp, cta, clin, aaa):
    _, logvar = model_no_comp(cta, clin, aaa)
    assert logvar.min().item() >= -4.0 - 1e-5
    assert logvar.max().item() <=  2.0 + 1e-5


def test_forward_no_comp_output_finite(model_no_comp, cta, clin, aaa):
    logit, logvar = model_no_comp(cta, clin, aaa)
    assert torch.isfinite(logit).all()
    assert torch.isfinite(logvar).all()


# ── forward — with complications variant ─────────────────────────────────────


def test_forward_with_comp_output_count(model_with_comp, cta, clin, aaa):
    out = model_with_comp(cta, clin, aaa)
    assert len(out) == 4, "with-comp model must return exactly 4 tensors"


def test_forward_with_comp_shapes(model_with_comp, cta, clin, aaa):
    logit, logvar, comp_logits, comp_logvar = model_with_comp(cta, clin, aaa)
    assert logit.shape      == (BATCH, 1)
    assert logvar.shape     == (BATCH, 1)
    assert comp_logits.shape == (BATCH, N_COMP)
    assert comp_logvar.shape == (BATCH, N_COMP)


def test_forward_with_comp_logvar_clamped(model_with_comp, cta, clin, aaa):
    _, logvar, _, comp_logvar = model_with_comp(cta, clin, aaa)
    assert logvar.min().item()      >= -4.0 - 1e-5
    assert logvar.max().item()      <=  2.0 + 1e-5
    assert comp_logvar.min().item() >= -4.0 - 1e-5
    assert comp_logvar.max().item() <=  2.0 + 1e-5


def test_forward_with_comp_all_finite(model_with_comp, cta, clin, aaa):
    for t in model_with_comp(cta, clin, aaa):
        assert torch.isfinite(t).all()


# ── return_fused flag ─────────────────────────────────────────────────────────


def test_return_fused_no_comp(model_no_comp, cta, clin, aaa):
    fused = model_no_comp(cta, clin, aaa, return_fused=True)
    assert isinstance(fused, torch.Tensor)
    assert fused.shape == (BATCH, 320)  # 256 + 32 + 32


def test_return_fused_with_comp(model_with_comp, cta, clin, aaa):
    fused = model_with_comp(cta, clin, aaa, return_fused=True)
    assert fused.shape == (BATCH, 320)


# ── model attributes ──────────────────────────────────────────────────────────


def test_no_comp_has_no_complications_head(model_no_comp):
    assert not hasattr(model_no_comp, "complications_head")


def test_with_comp_has_complications_head(model_with_comp):
    assert hasattr(model_with_comp, "complications_head")


def test_complications_stored_on_model(model_with_comp):
    assert model_with_comp.complications == COMPLICATIONS


def test_feature_gate_shape(model_no_comp):
    assert model_no_comp.feature_gate.shape == (CLINICAL_DIM,)


# ── load_checkpoint ───────────────────────────────────────────────────────────


def test_load_checkpoint_basic(tmp_path, fresh_model_no_comp):
    from src.models.reintervention import load_checkpoint
    ckpt_path = tmp_path / "model.pt"
    torch.save({"state_dict": fresh_model_no_comp.state_dict(), "threshold": 0.42}, ckpt_path)

    fresh_model_no_comp2 = type(fresh_model_no_comp).__new__(type(fresh_model_no_comp))
    # Just load into same model (weights already match)
    ckpt = load_checkpoint(fresh_model_no_comp, str(ckpt_path), device="cpu")
    assert "threshold" in ckpt
    assert ckpt["threshold"] == 0.42


def test_load_checkpoint_strips_dataparallel_prefix(tmp_path, fresh_model_no_comp):
    from src.models.reintervention import load_checkpoint
    # Simulate a checkpoint saved from DataParallel (all keys prefixed with "module.")
    dp_state = {"module." + k: v for k, v in fresh_model_no_comp.state_dict().items()}
    ckpt_path = tmp_path / "dp_model.pt"
    torch.save(dp_state, ckpt_path)

    ckpt = load_checkpoint(fresh_model_no_comp, str(ckpt_path), device="cpu")
    # Should not raise; model state dict has no module. prefix
    assert ckpt is not None


def test_load_checkpoint_filters_unexpected_keys(tmp_path, fresh_model_no_comp):
    from src.models.reintervention import load_checkpoint
    state = fresh_model_no_comp.state_dict()
    state["clinical_head.weight"] = torch.zeros(10, 10)  # legacy key
    state["clinical_head.bias"]   = torch.zeros(10)
    ckpt_path = tmp_path / "legacy.pt"
    torch.save(state, ckpt_path)

    # strict=False should silently drop unexpected keys
    ckpt = load_checkpoint(fresh_model_no_comp, str(ckpt_path), device="cpu", strict=False)
    assert ckpt is not None


def test_load_checkpoint_strict_raises_on_missing(tmp_path, fresh_model_no_comp):
    from src.models.reintervention import load_checkpoint
    # Save only a partial state dict
    partial = {"feature_gate": fresh_model_no_comp.feature_gate.data}
    ckpt_path = tmp_path / "partial.pt"
    torch.save(partial, ckpt_path)

    with pytest.raises(RuntimeError, match="Missing"):
        load_checkpoint(fresh_model_no_comp, str(ckpt_path), device="cpu", strict=True)


# ── FiLM clamp (±0.15) ───────────────────────────────────────────────────────
# _film(): x * (1 + tanh(γ) * 0.15) + tanh(β) * 0.15
# tanh saturates to ±1, so the multiplicative scale stays in [0.85, 1.15]
# and the additive shift stays in [-0.15, 0.15].


def test_film_multiplicative_scale_clamped(fresh_model_no_comp):
    """With extreme γ, the multiplicative factor saturates to exactly 1.15 (tanh→1)."""
    fusion_dim = 320
    x          = torch.rand(BATCH, fusion_dim)
    gamma_beta = torch.full((BATCH, fusion_dim * 2), 1e6)   # saturates tanh to +1

    result   = fresh_model_no_comp._film(x, gamma_beta)
    expected = x * 1.15 + 0.15                              # scale=1.15, shift=+0.15
    assert torch.allclose(result, expected, atol=1e-4)


def test_film_additive_shift_positive_bound(fresh_model_no_comp):
    """Additive shift with extreme positive β and zero input must equal +0.15."""
    fusion_dim = 320
    x          = torch.zeros(BATCH, fusion_dim)
    gamma_beta = torch.full((BATCH, fusion_dim * 2), 1e6)

    result = fresh_model_no_comp._film(x, gamma_beta)
    assert torch.allclose(result, torch.full_like(result, 0.15), atol=1e-4)


def test_film_additive_shift_negative_bound(fresh_model_no_comp):
    """Additive shift with extreme negative β and zero input must equal -0.15."""
    fusion_dim = 320
    x          = torch.zeros(BATCH, fusion_dim)
    gamma_beta = torch.full((BATCH, fusion_dim * 2), -1e6)  # tanh→-1 → shift=-0.15

    result = fresh_model_no_comp._film(x, gamma_beta)
    assert torch.allclose(result, torch.full_like(result, -0.15), atol=1e-4)
