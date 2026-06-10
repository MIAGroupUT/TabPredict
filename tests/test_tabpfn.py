"""
Tests for src/models/tabpfn.py — TabPFNBranchWrapper.

All tests run on CPU with synthetic data.  No TabPFN fitting is performed
(weight download + fine-tuning is slow); tests cover structural correctness,
the differentiable/manages_own_grad markers, the zero-before-fit guard, and
full-model integration via is_classical_backbone().
"""

from __future__ import annotations

import torch
import pytest

from tests.conftest import AAA_DIM, CLINICAL_DIM, BATCH

N_FEAT = CLINICAL_DIM + AAA_DIM


# ── TabPFNBranchWrapper structural tests ──────────────────────────────────────

def test_tabpfn_backbone_builds():
    from src.models.tabpfn import TabPFNBranchWrapper
    w = TabPFNBranchWrapper(in_features=N_FEAT, epochs=1, n_estimators_final_inference=1)
    assert w is not None
    assert not w._fitted


def test_tabpfn_backbone_is_differentiable():
    from src.models.tabpfn import TabPFNBranchWrapper
    assert TabPFNBranchWrapper._is_non_differentiable is False
    assert TabPFNBranchWrapper._manages_own_grad is True


@pytest.mark.parametrize("n_estimators", [1, 4])
def test_tabpfn_projection_width_equals_n_estimators_times_two(n_estimators):
    """projection input = n_estimators * 2 (per-estimator raw logits for both classes)."""
    from src.models.tabpfn import TabPFNBranchWrapper
    w = TabPFNBranchWrapper(in_features=N_FEAT, n_estimators_final_inference=n_estimators,
                            epochs=1)
    assert w.projection.in_features == n_estimators * 2
    assert w.projection.out_features == 32


def test_tabpfn_stores_n_estimators():
    from src.models.tabpfn import TabPFNBranchWrapper
    w = TabPFNBranchWrapper(in_features=N_FEAT, n_estimators_final_inference=4, epochs=1)
    assert w.n_estimators == 4


def test_tabpfn_backbone_returns_zeros_before_fit():
    from src.models.tabpfn import TabPFNBranchWrapper
    w = TabPFNBranchWrapper(in_features=N_FEAT, n_estimators_final_inference=4, epochs=1)
    x = torch.rand(BATCH, N_FEAT)
    out = w(x)
    assert out.shape == (BATCH, 32)
    assert out.sum().item() == pytest.approx(0.0)



# ── ReinterventionRiskModel integration ──────────────────────────────────────

def test_tabpfn_model_builds():
    from src.models.reintervention import ReinterventionRiskModel
    m = ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[],
        clinical_backbone="tabpfn",
        tabm_cfg=dict(epochs=1, n_estimators_final_inference=1),
    )
    assert m is not None


def test_tabpfn_is_not_classical_backbone():
    """TabPFN is now differentiable — it should not be treated as a classical backbone."""
    from src.models.reintervention import ReinterventionRiskModel, is_classical_backbone
    m = ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[],
        clinical_backbone="tabpfn",
        tabm_cfg=dict(epochs=1, n_estimators_final_inference=1),
    )
    assert not is_classical_backbone(m)


def test_tabpfn_not_in_classical_backbones_list():
    """tabpfn is handled via its own branch in _build_clinical_branch, not
    via the sklearn CLASSICAL_BACKBONES list."""
    from src.models.baselines import CLASSICAL_BACKBONES
    assert "tabpfn" not in CLASSICAL_BACKBONES


def test_tabpfn_model_stores_backbone_name():
    from src.models.reintervention import ReinterventionRiskModel
    m = ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[],
        clinical_backbone="tabpfn",
        tabm_cfg=dict(epochs=1, n_estimators_final_inference=1),
    )
    assert m.clinical_backbone == "tabpfn"


def test_tabpfn_model_branch_type():
    from src.models.reintervention import ReinterventionRiskModel
    from src.models.tabpfn import TabPFNBranchWrapper
    m = ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[],
        clinical_backbone="tabpfn",
        tabm_cfg=dict(epochs=1, n_estimators_final_inference=1),
    )
    assert isinstance(m.clinical_branch, TabPFNBranchWrapper)


# ── is_classical_backbone still true for sklearn backends ────────────────────

def test_sklearn_backbone_still_classical():
    """Refactoring must not break is_classical_backbone for existing sklearn backends."""
    from src.models.reintervention import ReinterventionRiskModel, is_classical_backbone
    m = ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[],
        clinical_backbone="lr",
    )
    assert is_classical_backbone(m)


def test_mlp_backbone_not_classical():
    from src.models.reintervention import ReinterventionRiskModel, is_classical_backbone
    m = ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[],
        clinical_backbone="mlp",
    )
    assert not is_classical_backbone(m)
