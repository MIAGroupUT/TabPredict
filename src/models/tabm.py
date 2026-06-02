"""
TabMEmbed — thin wrapper around the official ``tabm`` package.

Satisfies the clinical-branch contract:
    forward(x: Tensor[B, clinical_dim]) → Tensor[B, 32]

Internally it builds a ``TabM`` ensemble of k MLPs (all numerical features,
no categoricals).  The official model returns ``(B, k, d_out)``; we reduce
over the k-axis via mean to get the deterministic 32-dim embedding that FiLM
fusion expects.  Member diversity is the regulariser.

Numerical feature embeddings (``PeriodicEmbeddings``) are enabled by default
(``num_embeddings='periodic'``).  Each scalar feature is first projected to a
``d_embedding``-dim sinusoidal representation before the MLP ensemble.  This
captures non-linear feature relationships without adding per-feature parameters
(``lite=True`` uses shared frequencies across all features — the safe choice
for small datasets).  Pass ``num_embeddings='none'`` to disable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tabm import TabM

__all__ = ["TabMEmbed"]


def _build_periodic_embeddings(n_features: int, d_embedding: int):
    from rtdl_num_embeddings import PeriodicEmbeddings
    # lite=True: shared frequencies across features — fewer params, less overfit risk
    return PeriodicEmbeddings(n_features=n_features, d_embedding=d_embedding, lite=True)


class TabMEmbed(nn.Module):
    """
    Wrap ``TabM`` so it acts as a drop-in clinical branch:
    ``(B, clinical_dim) → (B, out_dim)`` via ensemble mean.

    Args:
        clinical_dim:    Number of (z-normalised) input features.
        out_dim:         Output embedding size (32 to match FiLM fusion).
        k:               Ensemble size.  Paper uses 32; 16 for small datasets.
        n_blocks:        Depth of each MLP in the ensemble.
        d_block:         Hidden width of each MLP block.
        dropout:         Dropout rate applied inside each block.
        arch_type:       ``'tabm'`` (default) | ``'tabm-mini'`` | ``'tabm-packed'``.
        activation:      Activation name recognised by TabM (default ``'ReLU'``).
        num_embeddings:  ``'periodic'`` (default) | ``'none'``.  Whether to apply
                         ``PeriodicEmbeddings`` (``lite=True``) to each input feature
                         before the MLP ensemble.
        d_embedding:     Embedding dimension per feature when ``num_embeddings='periodic'``.
    """

    def __init__(
        self,
        clinical_dim: int,
        out_dim: int = 32,
        k: int = 16,
        n_blocks: int = 2,
        d_block: int = 128,
        dropout: float = 0.3,
        arch_type: str = "tabm",
        activation: str = "ReLU",
        num_embeddings: str = "periodic",
        d_embedding: int = 16,
    ) -> None:
        super().__init__()
        emb = (
            _build_periodic_embeddings(clinical_dim, d_embedding)
            if num_embeddings == "periodic"
            else None
        )
        self.tabm = TabM.make(
            n_num_features=clinical_dim,
            cat_cardinalities=[],
            d_out=out_dim,
            num_embeddings=emb,
            k=k,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            arch_type=arch_type,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, clinical_dim)`` gated clinical features.
        Returns:
            ``(B, out_dim)`` embedding (mean over ensemble members).
        """
        # TabM returns (B, k, out_dim); mean over k → (B, out_dim)
        return self.tabm(x, None).mean(dim=1)
