"""
Shared fixtures and heavy-dependency mocks for the TabPredict test suite.

merlin is patched into sys.modules BEFORE any src.* import so the
test process never needs that package installed.  All tests run on CPU only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── add repo root to path so `from src.*` works ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── stub out merlin + peft BEFORE any src import ──────────────────────────────


class _MockMerlin(nn.Module):
    """Lightweight stand-in for the 3D ViT backbone (Merlin).

    Returns a (B, 2048) zero tensor regardless of input shape so the model
    architecture wires up correctly without any pretrained weights.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 2048)


_merlin_mod = MagicMock()
_merlin_mod.Merlin = _MockMerlin
sys.modules.setdefault("merlin", _merlin_mod)

# ── shared dimension constants ─────────────────────────────────────────────────

CLINICAL_DIM = 8   # real = 744; tiny for speed
AAA_DIM = 5
COMPLICATIONS = ["endoleak_1a", "endoleak_2", "migration"]
N_COMP = len(COMPLICATIONS)
BATCH = 2
SPATIAL = 6        # tiny spatial dims for CPU speed

# ── session-scoped model fixtures (eval only, never mutated) ──────────────────


@pytest.fixture(scope="session")
def model_no_comp():
    from src.models.reintervention import ReinterventionRiskModel
    return ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[]    ).eval()


@pytest.fixture(scope="session")
def model_with_comp():
    from src.models.reintervention import ReinterventionRiskModel
    return ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM,
        complications=COMPLICATIONS    ).eval()


# ── function-scoped model fixtures (safe to mutate in stage/optimizer tests) ──


@pytest.fixture
def fresh_model_no_comp():
    from src.models.reintervention import ReinterventionRiskModel
    return ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM, complications=[]    )


@pytest.fixture
def fresh_model_with_comp():
    from src.models.reintervention import ReinterventionRiskModel
    return ReinterventionRiskModel(
        clinical_dim=CLINICAL_DIM, aaa_dim=AAA_DIM,
        complications=COMPLICATIONS    )


# ── standard tensor fixtures ──────────────────────────────────────────────────


@pytest.fixture
def cta():
    return torch.rand(BATCH, 4, SPATIAL, SPATIAL, SPATIAL)


@pytest.fixture
def clin():
    return torch.rand(BATCH, CLINICAL_DIM)


@pytest.fixture
def aaa():
    return torch.rand(BATCH, AAA_DIM)


@pytest.fixture
def reint_gt():
    """Mixed 0/1 labels, shape (B, 1)."""
    return torch.FloatTensor([0.0, 1.0]).unsqueeze(1)


@pytest.fixture
def comp_gt():
    """Mix of 0/1/-1 (missing) labels, shape (B, N_COMP)."""
    return torch.FloatTensor([
        [1.0,  0.0, -1.0],
        [0.0,  1.0,  1.0],
    ])
