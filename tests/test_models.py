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
    assert len(out) == 1, "no-comp model must return exactly 1 tensor"


def test_forward_no_comp_shapes(model_no_comp, cta, clin, aaa):
    (logit,) = model_no_comp(cta, clin, aaa)
    assert logit.shape == (BATCH, 1)


def test_forward_no_comp_output_finite(model_no_comp, cta, clin, aaa):
    (logit,) = model_no_comp(cta, clin, aaa)
    assert torch.isfinite(logit).all()


# ── forward — with complications variant ─────────────────────────────────────


def test_forward_with_comp_output_count(model_with_comp, cta, clin, aaa):
    out = model_with_comp(cta, clin, aaa)
    assert len(out) == 2, "with-comp model must return exactly 2 tensors"


def test_forward_with_comp_shapes(model_with_comp, cta, clin, aaa):
    logit, comp_logits = model_with_comp(cta, clin, aaa)
    assert logit.shape       == (BATCH, 1)
    assert comp_logits.shape == (BATCH, N_COMP)


def test_forward_with_comp_all_finite(model_with_comp, cta, clin, aaa):
    for t in model_with_comp(cta, clin, aaa):
        assert torch.isfinite(t).all()


# ── return_fused flag ─────────────────────────────────────────────────────────


def test_return_fused_no_comp(model_no_comp, cta, clin, aaa):
    fused = model_no_comp(cta, clin, aaa, return_fused=True)
    assert isinstance(fused, torch.Tensor)
    assert fused.shape == (BATCH, 288)  # 256 + 32


def test_return_fused_with_comp(model_with_comp, cta, clin, aaa):
    fused = model_with_comp(cta, clin, aaa, return_fused=True)
    assert fused.shape == (BATCH, 288)


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


# ── FiLM clamp (±0.3) ────────────────────────────────────────────────────────
# _film(): x * (1 + tanh(γ) * 0.3) + tanh(β) * 0.3
# tanh saturates to ±1, so the multiplicative scale stays in [0.7, 1.3]
# and the additive shift stays in [-0.3, 0.3].


def test_film_multiplicative_scale_clamped(fresh_model_no_comp):
    """With extreme γ, the multiplicative factor saturates to exactly 1.3 (tanh→1)."""
    fusion_dim = 288
    x          = torch.rand(BATCH, fusion_dim)
    gamma_beta = torch.full((BATCH, fusion_dim * 2), 1e6)   # saturates tanh to +1

    result   = fresh_model_no_comp._film(x, gamma_beta)
    expected = x * 1.3 + 0.3                                # scale=1.3, shift=+0.3
    assert torch.allclose(result, expected, atol=1e-4)


def test_film_additive_shift_positive_bound(fresh_model_no_comp):
    """Additive shift with extreme positive β and zero input must equal +0.3."""
    fusion_dim = 288
    x          = torch.zeros(BATCH, fusion_dim)
    gamma_beta = torch.full((BATCH, fusion_dim * 2), 1e6)

    result = fresh_model_no_comp._film(x, gamma_beta)
    assert torch.allclose(result, torch.full_like(result, 0.3), atol=1e-4)


def test_film_additive_shift_negative_bound(fresh_model_no_comp):
    """Additive shift with extreme negative β and zero input must equal -0.3."""
    fusion_dim = 288
    x          = torch.zeros(BATCH, fusion_dim)
    gamma_beta = torch.full((BATCH, fusion_dim * 2), -1e6)  # tanh→-1 → shift=-0.3

    result = fresh_model_no_comp._film(x, gamma_beta)
    assert torch.allclose(result, torch.full_like(result, -0.3), atol=1e-4)


# ── TabM clinical backbone ────────────────────────────────────────────────────

_TABM_CFG = dict(k=4, n_blocks=1, d_block=32, dropout=0.0, num_embeddings="periodic", d_embedding=8)


@pytest.fixture(scope="module")
def model_tabm_no_comp():
    from src.models.reintervention import ReinterventionRiskModel
    return ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[],
        clinical_backbone="tabm", tabm_cfg=_TABM_CFG,
    ).eval()


@pytest.fixture(scope="module")
def model_tabm_with_comp():
    from src.models.reintervention import ReinterventionRiskModel
    return ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=COMPLICATIONS,
        clinical_backbone="tabm", tabm_cfg=_TABM_CFG,
    ).eval()


def test_tabm_model_builds(model_tabm_no_comp):
    assert model_tabm_no_comp is not None


def test_tabm_model_stores_backbone(model_tabm_no_comp):
    assert model_tabm_no_comp.clinical_backbone == "tabm"


def test_tabm_forward_no_comp_shape(model_tabm_no_comp, cta, clin, aaa):
    out = model_tabm_no_comp(cta, clin, aaa)
    assert len(out) == 1
    assert out[0].shape == (BATCH, 1)


def test_tabm_forward_with_comp_shape(model_tabm_with_comp, cta, clin, aaa):
    out = model_tabm_with_comp(cta, clin, aaa)
    assert len(out) == 2
    assert out[0].shape == (BATCH, 1)
    assert out[1].shape == (BATCH, N_COMP)


def test_tabm_forward_finite(model_tabm_no_comp, cta, clin, aaa):
    out = model_tabm_no_comp(cta, clin, aaa)
    assert torch.isfinite(out[0]).all()


def test_tabm_clinical_branch_output_dim(model_tabm_no_comp, clin, aaa):
    """clinical_branch must output (B, 32) regardless of backbone."""
    gated = clin * torch.sigmoid(model_tabm_no_comp.feature_gate)
    c_in  = torch.cat([gated, aaa], dim=1)
    emb   = model_tabm_no_comp.clinical_branch(c_in)
    assert emb.shape == (BATCH, 32)


def test_unknown_backbone_raises():
    from src.models.reintervention import ReinterventionRiskModel
    with pytest.raises(ValueError, match="Unknown clinical_backbone"):
        ReinterventionRiskModel(
            clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM,
            clinical_backbone="unknown",
        )
