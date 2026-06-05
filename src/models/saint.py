"""
SAINTEmbed — SAINT (Self-Attention and Intersample Attention Transformer)
adapted as a drop-in tabular branch for ReinterventionRiskModel.

Source: https://github.com/somepago/saint  (MIT licence)

Adaptations vs. the official code
----------------------------------
* All-continuous input only (no categorical features / embedding tables).
* Per-feature MLP embedder: simple_MLP([1, mlp_hidden, dim]) — faithful to SAINT.
* Column attention: standard self-attention over feature tokens (always present).
* Row attention: intersample attention over batch (only when style='colrow').
  Row FF is a bottleneck MLP (dim → row_ff_hidden → dim) rather than the upstream
  GEGLU with mult=4, which explodes for large n_features (e.g. 47×16=752 → 1.7M
  params per layer with the original FF).  Set row_ff_hidden to control the size.
* Output: flatten feature tokens → Linear → (B, out_dim=32).
  No classification head — that lives in ReinterventionRiskModel.

Contract
--------
    forward(x: Tensor[B, clinical_dim]) → Tensor[B, out_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["SAINTEmbed"]


# ── SAINT primitives (from somepago/saint models/model.py) ───────────────────

class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 16,
        dropout: float = 0.,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h=h) for t in (q, k, v))
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.dropout(sim.softmax(dim=-1))
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)', h=h))


class simple_MLP(nn.Module):
    """Two-layer MLP with ReLU (faithful to the official SAINT implementation)."""
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.layers(x)


class BottleneckFF(nn.Module):
    """Lightweight FF for row attention: dim → hidden → dim (ReLU, no GEGLU).
    Avoids the O(dim²) parameter explosion of the upstream GEGLU FF when
    dim = n_features × d_embed is large (~700+ for 47 features × dim=16)."""
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x)


# ── SAINTEmbed ────────────────────────────────────────────────────────────────

class SAINTEmbed(nn.Module):
    """
    SAINT tabular backbone for all-continuous features.

    Args:
        clinical_dim:   Number of input features (gated clinical + AAA scalars).
        out_dim:        Output embedding size (32 to match FiLM fusion).
        dim:            Per-feature embedding dimension.
        depth:          Number of transformer layers.
        heads:          Number of attention heads (column attention).
        dim_head:       Head dimension (column attention).
        attn_dropout:   Attention dropout.
        ff_dropout:     Feed-forward dropout.
        style:          ``'col'`` (column attention only, i.e. SAINT-s)
                        or ``'colrow'`` (column + row / intersample attention).
        mlp_hidden:     Hidden size of per-feature embedding MLP.
        row_heads:      Heads for row (intersample) attention.  Defaults to
                        ``heads`` when not set.
        row_dim_head:   Head dim for row attention.
        row_ff_hidden:  Bottleneck hidden size for row FF (default 128).
                        Replaces the upstream GEGLU FF to prevent param explosion
                        when n_features is large (47 features × dim=16 → 752-dim rows).
    """

    def __init__(
        self,
        clinical_dim: int,
        out_dim: int = 32,
        dim: int = 16,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        style: str = "col",
        mlp_hidden: int = 32,
        row_heads: int | None = None,
        row_dim_head: int = 16,
        row_ff_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.clinical_dim = clinical_dim
        self.dim = dim
        self.style = style

        # Per-feature continuous embedding (one small MLP per feature).
        # Faithful to SAINT: simple_MLP([1, mlp_hidden, dim]).
        self.feature_embeds = nn.ModuleList([
            simple_MLP([1, mlp_hidden, dim]) for _ in range(clinical_dim)
        ])

        _row_heads = row_heads if row_heads is not None else heads
        row_dim = dim * clinical_dim

        # Build depth transformer layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.ModuleList([
                # column attention
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, mult=4, dropout=ff_dropout))),
            ])
            if style == "colrow":
                layer.append(PreNorm(row_dim, Residual(Attention(row_dim, heads=_row_heads, dim_head=row_dim_head, dropout=attn_dropout))))
                layer.append(PreNorm(row_dim, Residual(BottleneckFF(row_dim, hidden=row_ff_hidden, dropout=ff_dropout))))
            self.layers.append(layer)

        # Final projection: flatten all feature tokens → out_dim
        self.proj = nn.Linear(row_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, clinical_dim)`` normalised features.
        Returns:
            ``(B, out_dim)`` embedding.
        """
        B, n = x.shape

        # Per-feature embedding: (B, n, dim)
        x_emb = torch.stack([
            self.feature_embeds[i](x[:, i:i+1])
            for i in range(n)
        ], dim=1)

        for layer in self.layers:
            if self.style == "colrow":
                col_attn, col_ff, row_attn, row_ff = layer
            else:
                col_attn, col_ff = layer

            # Column attention: each sample processes its own feature tokens
            x_emb = col_attn(x_emb)
            x_emb = col_ff(x_emb)

            if self.style == "colrow":
                # Row (intersample) attention: flatten per-sample tokens,
                # then attend across samples in the batch.
                x_flat = rearrange(x_emb, 'b n d -> 1 b (n d)')
                x_flat = row_attn(x_flat)
                x_flat = row_ff(x_flat)
                x_emb = rearrange(x_flat, '1 b (n d) -> b n d', n=n)

        return self.proj(x_emb.flatten(1))   # (B, out_dim)
