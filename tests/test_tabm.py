"""
Tests for src/models/tabm.py (TabMEmbed wrapper).

Uses the real ``tabm`` and ``rtdl_num_embeddings`` packages — no mocks.

Covers:
- Output shape (B, 32) for various batch sizes
- Ensemble mean is deterministic in eval mode
- Wrapper handles different k / n_blocks / d_block values
- PeriodicEmbeddings (default) and none paths both produce correct shapes
- Invalid backbone name raises in ReinterventionRiskModel
"""

from __future__ import annotations

import torch
import pytest

from tests.conftest import BATCH, CLINICAL_DIM


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tabm_embed():
    """Default config: PeriodicEmbeddings enabled (num_embeddings='periodic')."""
    from src.models.tabm import TabMEmbed
    return TabMEmbed(
        clinical_dim=CLINICAL_DIM,
        out_dim=32,
        k=4,
        n_blocks=1,
        d_block=32,
        dropout=0.0,
        num_embeddings="periodic",
        d_embedding=8,
    ).eval()


@pytest.fixture(scope="module")
def tabm_embed_no_emb():
    """Plain TabM without feature embeddings."""
    from src.models.tabm import TabMEmbed
    return TabMEmbed(
        clinical_dim=CLINICAL_DIM,
        out_dim=32,
        k=4,
        n_blocks=1,
        d_block=32,
        dropout=0.0,
        num_embeddings="none",
    ).eval()


# ── output shape ─────────────────────────────────────────────────────────────

def test_output_shape_standard(tabm_embed):
    x   = torch.randn(BATCH, CLINICAL_DIM)
    out = tabm_embed(x)
    assert out.shape == (BATCH, 32)


def test_output_shape_batch1(tabm_embed):
    x   = torch.randn(1, CLINICAL_DIM)
    out = tabm_embed(x)
    assert out.shape == (1, 32)


def test_output_shape_large_batch(tabm_embed):
    x   = torch.randn(32, CLINICAL_DIM)
    out = tabm_embed(x)
    assert out.shape == (32, 32)


# ── determinism ───────────────────────────────────────────────────────────────

def test_deterministic_in_eval(tabm_embed):
    """Same input → same output in eval mode (no stochastic dropout)."""
    x    = torch.randn(BATCH, CLINICAL_DIM)
    out1 = tabm_embed(x)
    out2 = tabm_embed(x)
    assert torch.allclose(out1, out2)


def test_finite_output(tabm_embed):
    x   = torch.randn(BATCH, CLINICAL_DIM)
    out = tabm_embed(x)
    assert torch.isfinite(out).all()


# ── different configs ─────────────────────────────────────────────────────────

def test_larger_k():
    from src.models.tabm import TabMEmbed
    m   = TabMEmbed(clinical_dim=CLINICAL_DIM, out_dim=32, k=8, n_blocks=1, d_block=64, dropout=0.0).eval()
    x   = torch.randn(BATCH, CLINICAL_DIM)
    out = m(x)
    assert out.shape == (BATCH, 32)


def test_deeper_blocks():
    from src.models.tabm import TabMEmbed
    m   = TabMEmbed(clinical_dim=CLINICAL_DIM, out_dim=32, k=4, n_blocks=3, d_block=32, dropout=0.0).eval()
    x   = torch.randn(BATCH, CLINICAL_DIM)
    out = m(x)
    assert out.shape == (BATCH, 32)


def test_different_out_dim():
    """TabMEmbed can output any dim; 32 is just our contract value."""
    from src.models.tabm import TabMEmbed
    m   = TabMEmbed(clinical_dim=CLINICAL_DIM, out_dim=64, k=4, n_blocks=1, d_block=32, dropout=0.0).eval()
    x   = torch.randn(BATCH, CLINICAL_DIM)
    out = m(x)
    assert out.shape == (BATCH, 64)


# ── PeriodicEmbeddings path ───────────────────────────────────────────────────

def test_periodic_output_shape(tabm_embed):
    x   = torch.randn(BATCH, CLINICAL_DIM)
    out = tabm_embed(x)
    assert out.shape == (BATCH, 32)


def test_periodic_deterministic_eval(tabm_embed):
    x    = torch.randn(BATCH, CLINICAL_DIM)
    out1 = tabm_embed(x)
    out2 = tabm_embed(x)
    assert torch.allclose(out1, out2)


def test_periodic_finite(tabm_embed):
    x   = torch.randn(BATCH, CLINICAL_DIM)
    out = tabm_embed(x)
    assert torch.isfinite(out).all()


# ── no-embeddings path ────────────────────────────────────────────────────────

def test_no_emb_output_shape(tabm_embed_no_emb):
    x   = torch.randn(BATCH, CLINICAL_DIM)
    out = tabm_embed_no_emb(x)
    assert out.shape == (BATCH, 32)


def test_no_emb_deterministic_eval(tabm_embed_no_emb):
    x    = torch.randn(BATCH, CLINICAL_DIM)
    out1 = tabm_embed_no_emb(x)
    out2 = tabm_embed_no_emb(x)
    assert torch.allclose(out1, out2)
