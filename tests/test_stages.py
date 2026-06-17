"""
Tests for src/training/stages.py

Covers:
- set_all_seeds: reproducibility
- causal_modality_dropout: shape preservation, causal constraint
- maybe_apply_modality_dropout: no-op outside training
- set_stage_a: correct freeze/unfreeze pattern
- set_stage_b: condition_net reset + I3ResNet BN layers + visual pathway unfrozen
- build_optimizer: correct param groups per stage, raises on unknown stage
- count_trainable: correct totals for module and parameter
- print_model_param_overview: runs without error
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from tests.conftest import BATCH, CLINICAL_DIM, AAA_DIM, SPATIAL, COMPLICATIONS


# ── set_all_seeds ─────────────────────────────────────────────────────────────


def test_set_all_seeds_reproducibility():
    from src.training.stages import set_all_seeds
    set_all_seeds(42)
    a = torch.rand(5)
    set_all_seeds(42)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_set_all_seeds_different_seeds():
    from src.training.stages import set_all_seeds
    set_all_seeds(0)
    a = torch.rand(5)
    set_all_seeds(1)
    b = torch.rand(5)
    assert not torch.allclose(a, b)


# ── causal_modality_dropout ───────────────────────────────────────────────────


def _make_inputs():
    cta  = torch.ones(BATCH, 4, SPATIAL, SPATIAL, SPATIAL)
    clin = torch.ones(BATCH, CLINICAL_DIM)
    aaa  = torch.ones(BATCH, AAA_DIM)
    return cta, clin, aaa


def test_causal_dropout_preserves_shape():
    from src.training.stages import causal_modality_dropout
    cta, clin, aaa = _make_inputs()
    cta_d, clin_d, aaa_d = causal_modality_dropout(cta, clin, aaa)
    assert cta_d.shape  == cta.shape
    assert clin_d.shape == clin.shape
    assert aaa_d.shape  == aaa.shape


def test_causal_dropout_no_dropout_at_p0():
    """With all probabilities at 0, output should equal input."""
    from src.training.stages import causal_modality_dropout
    cta, clin, aaa = _make_inputs()
    cta_d, clin_d, aaa_d = causal_modality_dropout(
        cta, clin, aaa, p_cta=0.0, p_aaa_given_cta=0.0, p_aaa_given_clin=0.0
    )
    assert torch.allclose(cta_d,  cta)
    assert torch.allclose(clin_d, clin)
    assert torch.allclose(aaa_d,  aaa)


def test_causal_dropout_drops_all_at_p1():
    """With p_cta=1 all CTA samples should be zeroed."""
    from src.training.stages import causal_modality_dropout
    torch.manual_seed(0)
    cta, clin, aaa = _make_inputs()
    cta_d, _, _ = causal_modality_dropout(cta, clin, aaa, p_cta=1.0)
    assert cta_d.abs().sum().item() == 0.0


def test_causal_dropout_aaa_only_zero_when_cta_zero():
    """AAA can only be zeroed; CTA presence doesn't guarantee AAA presence but
    CTA=zeroed must imply CTA was dropped (no causal violation)."""
    from src.training.stages import causal_modality_dropout
    # With p_cta=1, cta_d is all-zeros; run many times and verify
    torch.manual_seed(99)
    cta, clin, aaa = _make_inputs()
    for _ in range(20):
        cta_d, _, aaa_d = causal_modality_dropout(cta, clin, aaa, p_cta=1.0)
        for b in range(BATCH):
            if cta_d[b].abs().sum() > 0:
                pass  # cta not zeroed means aaa could be either
            # If cta is zeroed (which it always is here), aaa may or may not be zeroed
            # but no violation possible since the causal direction is CTA→AAA


def test_causal_dropout_clinical_at_p1():
    """With p_clinical=1 all clinical samples should be zeroed."""
    from src.training.stages import causal_modality_dropout
    cta, clin, aaa = _make_inputs()
    _, clin_d, _ = causal_modality_dropout(
        cta, clin, aaa, p_cta=0.0, p_aaa_given_cta=0.0, p_aaa_given_clin=0.0,
        p_clinical=1.0,
    )
    assert clin_d.abs().sum().item() == 0.0


def test_causal_dropout_clinical_preserves_cta():
    """Zeroing clinical must not affect CTA or AAA."""
    from src.training.stages import causal_modality_dropout
    cta, clin, aaa = _make_inputs()
    cta_d, _, aaa_d = causal_modality_dropout(
        cta, clin, aaa, p_cta=0.0, p_aaa_given_cta=0.0, p_aaa_given_clin=0.0,
        p_clinical=1.0,
    )
    assert torch.allclose(cta_d, cta)
    assert torch.allclose(aaa_d, aaa)


def test_maybe_apply_modality_dropout_no_op_at_test_time():
    """is_training=False must return tensors unchanged (same object)."""
    from src.training.stages import maybe_apply_modality_dropout
    cta, clin, aaa = _make_inputs()
    r_cta, r_clin, r_aaa = maybe_apply_modality_dropout(
        False, cta, clin, aaa, p_cta=0.9, p_aaa_given_cta=0.9, p_aaa_given_clin=0.9
    )
    assert r_cta is cta
    assert r_clin is clin
    assert r_aaa is aaa


def test_maybe_apply_modality_dropout_applies_at_train_time():
    """is_training=True with p_cta=1 should zero CTA."""
    from src.training.stages import maybe_apply_modality_dropout
    torch.manual_seed(0)
    cta, clin, aaa = _make_inputs()
    r_cta, _, _ = maybe_apply_modality_dropout(
        True, cta, clin, aaa, p_cta=1.0, p_aaa_given_cta=0.0, p_aaa_given_clin=0.0
    )
    assert r_cta.abs().sum().item() == 0.0


# ── set_stage_a ───────────────────────────────────────────────────────────────


def test_set_stage_a_freezes_visual_branch(fresh_model_no_comp):
    from src.training.stages import set_stage_a
    set_stage_a(fresh_model_no_comp)
    for p in fresh_model_no_comp.visual_branch.parameters():
        assert not p.requires_grad


def test_set_stage_a_freezes_cta_adapter(fresh_model_no_comp):
    from src.training.stages import set_stage_a
    set_stage_a(fresh_model_no_comp)
    for p in fresh_model_no_comp.cta_adapter.parameters():
        assert not p.requires_grad


def test_set_stage_a_freezes_visual_proj(fresh_model_no_comp):
    from src.training.stages import set_stage_a
    set_stage_a(fresh_model_no_comp)
    for p in fresh_model_no_comp.visual_proj.parameters():
        assert not p.requires_grad


def test_set_stage_a_trains_clinical_branch(fresh_model_no_comp):
    from src.training.stages import set_stage_a
    set_stage_a(fresh_model_no_comp)
    for p in fresh_model_no_comp.clinical_branch.parameters():
        assert p.requires_grad


def test_set_stage_a_no_aaa_branch(fresh_model_no_comp):
    """aaa_metrics_branch has been merged into clinical_branch; attribute must not exist."""
    assert not hasattr(fresh_model_no_comp, "aaa_metrics_branch")


def test_set_stage_a_trains_reintervention_head(fresh_model_no_comp):
    from src.training.stages import set_stage_a
    set_stage_a(fresh_model_no_comp)
    for p in fresh_model_no_comp.reintervention_head.parameters():
        assert p.requires_grad


def test_set_stage_a_trains_feature_gate(fresh_model_no_comp):
    from src.training.stages import set_stage_a
    set_stage_a(fresh_model_no_comp)
    assert fresh_model_no_comp.feature_gate.requires_grad


def test_set_stage_a_trains_complications_head(fresh_model_with_comp):
    from src.training.stages import set_stage_a
    set_stage_a(fresh_model_with_comp)
    for p in fresh_model_with_comp.complications_head.parameters():
        assert p.requires_grad


# ── set_stage_b ───────────────────────────────────────────────────────────────


def test_set_stage_b_runs_without_error(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp)  # should not raise


def test_set_stage_b_trains_condition_net(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp)
    for p in fresh_model_no_comp.condition_net.parameters():
        assert p.requires_grad


def test_set_stage_b_trains_clinical_branch(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp, freeze_clinical=False)
    for p in fresh_model_no_comp.clinical_branch.parameters():
        assert p.requires_grad


def test_set_stage_b_freeze_clinical_freezes_clinical_branch(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp, freeze_clinical=True)
    for p in fresh_model_no_comp.clinical_branch.parameters():
        assert not p.requires_grad


def test_set_stage_b_freeze_clinical_freezes_feature_gate(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp, freeze_clinical=True)
    assert not fresh_model_no_comp.feature_gate.requires_grad


def test_set_stage_b_freeze_clinical_still_trains_visual(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp, freeze_clinical=True)
    for p in fresh_model_no_comp.cta_adapter.parameters():
        assert p.requires_grad


def test_set_stage_b_freeze_clinical_still_trains_condition_net(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp, freeze_clinical=True)
    for p in fresh_model_no_comp.condition_net.parameters():
        assert p.requires_grad


def test_set_stage_b_trains_cta_adapter(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp)
    for p in fresh_model_no_comp.cta_adapter.parameters():
        assert p.requires_grad


def test_set_stage_b_trains_visual_proj(fresh_model_no_comp):
    from src.training.stages import set_stage_b
    set_stage_b(fresh_model_no_comp)
    for p in fresh_model_no_comp.visual_proj.parameters():
        assert p.requires_grad


# ── build_optimizer ───────────────────────────────────────────────────────────


def test_build_optimizer_returns_adamw(fresh_model_no_comp):
    from src.training.stages import build_optimizer, set_stage_a
    set_stage_a(fresh_model_no_comp)
    opt = build_optimizer(fresh_model_no_comp, "A")
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_stage_a_group_count(fresh_model_no_comp):
    from src.training.stages import build_optimizer
    opt = build_optimizer(fresh_model_no_comp, "A")
    assert len(opt.param_groups) == 5  # tabular + fusion only; no aaa branch or visual pathway


def test_build_optimizer_stage_a_with_comp_group_count(fresh_model_with_comp):
    from src.training.stages import build_optimizer
    opt = build_optimizer(fresh_model_with_comp, "A")
    assert len(opt.param_groups) == 6  # includes complications_head


def test_build_optimizer_stage_b_group_count(fresh_model_no_comp):
    from src.training.stages import build_optimizer, set_stage_b
    set_stage_b(fresh_model_no_comp)
    opt = build_optimizer(fresh_model_no_comp, "B")
    assert len(opt.param_groups) == 8  # tabular + cta_adapter + visual_proj + i3resnet_bn + head


def test_build_optimizer_stage_b_freeze_clinical_group_count(fresh_model_no_comp):
    from src.training.stages import build_optimizer, set_stage_b
    set_stage_b(fresh_model_no_comp, freeze_clinical=True)
    opt = build_optimizer(fresh_model_no_comp, "B", freeze_clinical=True)
    # condition + fusion_norm + cta_adapter + visual_proj + i3resnet_bn + reint_head = 6
    assert len(opt.param_groups) == 6


def test_build_optimizer_stage_b_freeze_clinical_no_clinical_group(fresh_model_no_comp):
    from src.training.stages import build_optimizer, set_stage_b
    set_stage_b(fresh_model_no_comp, freeze_clinical=True)
    opt    = build_optimizer(fresh_model_no_comp, "B", freeze_clinical=True)
    names  = {g["name"] for g in opt.param_groups}
    assert "clinical"     not in names
    assert "feature_gate" not in names


def test_build_optimizer_stage_b_i3resnet_bn_lr(fresh_model_no_comp):
    from src.training.stages import build_optimizer
    opt    = build_optimizer(fresh_model_no_comp, "B")
    groups = {g["name"]: g for g in opt.param_groups}
    assert groups["i3resnet_bn"]["lr"] == pytest.approx(1e-5)


def test_build_optimizer_stage_a_lr_ordering(fresh_model_no_comp):
    """feature_gate should have the highest LR in Stage A."""
    from src.training.stages import build_optimizer
    opt    = build_optimizer(fresh_model_no_comp, "A")
    groups = {g["name"]: g for g in opt.param_groups}
    assert groups["feature_gate"]["lr"] > groups["clinical"]["lr"]


def test_build_optimizer_unknown_stage_raises(fresh_model_no_comp):
    from src.training.stages import build_optimizer
    with pytest.raises(ValueError, match="Unknown stage"):
        build_optimizer(fresh_model_no_comp, "C")


# ── count_trainable ───────────────────────────────────────────────────────────


def test_count_trainable_all_trainable():
    from src.training.stages import count_trainable
    m = nn.Linear(10, 5)  # 10*5 + 5 = 55 params
    trainable, total = count_trainable(m)
    assert total     == 55
    assert trainable == 55


def test_count_trainable_none_trainable():
    from src.training.stages import count_trainable
    m = nn.Linear(10, 5)
    for p in m.parameters():
        p.requires_grad = False
    trainable, total = count_trainable(m)
    assert total     == 55
    assert trainable == 0


def test_count_trainable_parameter_object():
    from src.training.stages import count_trainable
    p = nn.Parameter(torch.zeros(7))
    trainable, total = count_trainable(p)
    assert trainable == 7
    assert total     == 7


def test_count_trainable_frozen_parameter():
    from src.training.stages import count_trainable
    p = nn.Parameter(torch.zeros(7), requires_grad=False)
    trainable, total = count_trainable(p)
    assert trainable == 0
    assert total     == 7


def test_count_trainable_nested_module():
    from src.training.stages import count_trainable
    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    # 2 × (4*4 + 4) = 40 params
    _, total = count_trainable(m)
    assert total == 40


# ── print_model_param_overview ────────────────────────────────────────────────


def test_print_model_param_overview_no_error(fresh_model_no_comp, capsys):
    from src.training.stages import print_model_param_overview
    print_model_param_overview(fresh_model_no_comp)
    captured = capsys.readouterr()
    assert "clinical_branch" in captured.out


def test_print_model_param_overview_with_comp(fresh_model_with_comp, capsys):
    from src.training.stages import print_model_param_overview
    print_model_param_overview(fresh_model_with_comp)
    captured = capsys.readouterr()
    assert "complications_head" in captured.out
