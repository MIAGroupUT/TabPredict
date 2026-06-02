"""
ReinterventionRiskModel — shared architecture for both training variants.

Instantiate with:
    complications=[]          → no complication head; forward returns (logit,)
    complications=[...]       → adds complications_head; forward returns (logit, comp_logits)

This single class replaces the two copies that lived in notebooks/.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from merlin import Merlin

# ── canonical complication lists (choose one per experiment config) ──────────

COMPLICATIONS_NO_COMP: list[str] = []  # primary-task-only variant

COMPLICATIONS_FULL: list[str] = [
    "endoleak_1a", "endoleak_1b", "endoleak_2", "endoleak_3", "endoleak_5",
    "migration", "infection",
]

AAA_METRIC_NAMES: list[str] = [
    "volume_lumen_ml",
    "volume_ilt_ml",
    "volume_total_ml",
    "max_lumen_to_wall_dist_mm",
    "avg_lumen_to_wall_dist_mm",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def get_model(m: nn.Module) -> nn.Module:
    """Unwrap nn.DataParallel."""
    return m.module if isinstance(m, nn.DataParallel) else m


def disable_checkpointing(model: nn.Module) -> None:
    """Patch gradient-checkpoint flags to avoid breaking GradCAM backward pass."""
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False
        if hasattr(m, "enable_gradient_checkpointing"):
            try:
                m.enable_gradient_checkpointing(False)
            except Exception:
                pass


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> dict:
    """
    Load a checkpoint produced by either the notebook training scripts or
    the new pipeline.  Handles:
      - DataParallel ``module.`` prefix stripping
      - Legacy keys not present in the current model (e.g. ``clinical_head``
        that existed as dead code in the no-complications notebook)

    Returns the full checkpoint dict so callers can read ``threshold``,
    ``feature_names``, ``clin_mean``, ``clin_std``, etc.
    """
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    # Strip DataParallel prefix if saved without get_model()
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(state.keys())

    unexpected = ckpt_keys - model_keys
    missing    = model_keys - ckpt_keys

    if unexpected:
        print(f"[load_checkpoint] Dropping {len(unexpected)} unexpected key(s): "
              f"{sorted(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing and strict:
        raise RuntimeError(f"Missing {len(missing)} key(s) in checkpoint: {sorted(missing)[:5]}")

    filtered_state = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(filtered_state, strict=False)
    return ckpt


# ── model ─────────────────────────────────────────────────────────────────────

class ReinterventionRiskModel(nn.Module):
    """
    Multi-modal reintervention risk predictor.

    Three input branches (CTA volume, clinical tabular, AAA geometry metrics)
    fused via FiLM conditioning, then decoded by task-specific heads.

    Args:
        clinical_dim:   Number of clinical tabular features (~744 in RADAR).
        aaa_dim:        Number of AAA geometry scalars (5).
        complications:  List of complication names.  Empty list → no
                        complications head; forward returns 2 tensors.
                        Non-empty → adds complications_head; forward returns 4.
    """

    def __init__(
        self,
        clinical_dim: int = 744,
        aaa_dim: int = 5,
        complications: list[str] | None = None,
        clinical_backbone: str = "mlp",
        tabm_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.complications = list(complications or [])
        self.clinical_backbone = clinical_backbone
        num_comp = len(self.complications)

        # ── visual pathway ────────────────────────────────────────────────
        self.cta_adapter = nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1), nn.ReLU(), nn.Conv3d(8, 1, 1)
        )
        self.visual_branch = Merlin(ImageEmbedding=True)
        disable_checkpointing(self.visual_branch)
        self.visual_proj = nn.Linear(2048, 256)

        # ── tabular pathways ─────────────────────────────────────────────
        # Learned input gate: sigmoid(gate) ∈ (0,1) per clinical feature only.
        # L1 reg during training pushes unimportant gates toward 0.
        # AAA scalars bypass the gate and are appended after gating.
        self.feature_gate = nn.Parameter(torch.zeros(clinical_dim))

        self.clinical_branch = self._build_clinical_branch(
            clinical_dim + aaa_dim, clinical_backbone, tabm_cfg or {}
        )

        # ── fusion (FiLM) ────────────────────────────────────────────────
        self._fusion_dim = 256 + 32  # 288
        self.condition_net = nn.Sequential(
            nn.Linear(self._fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2 * self._fusion_dim),
        )
        self.fusion_norm = nn.LayerNorm(self._fusion_dim)

        # ── output heads ─────────────────────────────────────────────────
        self.reintervention_head = nn.Sequential(
            nn.Linear(self._fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # logit only
        )
        if self.complications:
            self.complications_head = nn.Sequential(
                nn.Linear(self._fusion_dim, 256), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_comp),  # one logit per complication
            )

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _build_clinical_branch(
        clinical_dim: int,
        backbone: str,
        tabm_cfg: dict,
    ) -> nn.Module:
        """Return the clinical encoder branch for the requested backbone.

        Both variants accept ``(B, clinical_dim)`` and return ``(B, 32)``.
        """
        if backbone == "tabm":
            from src.models.tabm import TabMEmbed
            return TabMEmbed(clinical_dim=clinical_dim, out_dim=32, **tabm_cfg)
        if backbone == "mlp":
            return nn.Sequential(
                nn.Linear(clinical_dim, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 32),
                nn.Dropout(0.2),
            )
        raise ValueError(
            f"Unknown clinical_backbone {backbone!r}. Choose 'mlp' or 'tabm'."
        )

    def _film(self, x: torch.Tensor, gamma_beta: torch.Tensor) -> torch.Tensor:
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return x * (1 + torch.tanh(gamma) * 0.3) + torch.tanh(beta) * 0.3

    def _encode_visual(self, cta: torch.Tensor) -> torch.Tensor:
        v = self.visual_branch(cta)
        if isinstance(v, (tuple, list)):
            v = v[0]
        if v.dim() == 3 and v.shape[0] == 1:
            v = v.squeeze(0)
        return v.flatten(1) if v.dim() > 2 else v

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        cta: torch.Tensor,
        clin: torch.Tensor,
        aaa: torch.Tensor,
        return_fused: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Returns:
            If complications=[]:  (reint_logit,)
            If complications=[…]: (reint_logit, comp_logits)
        """
        cta   = self.cta_adapter(cta)
        v_emb = self.visual_proj(self._encode_visual(cta))
        c_in  = torch.cat([clin * torch.sigmoid(self.feature_gate), aaa], dim=1)
        c_emb = self.clinical_branch(c_in)

        fused = torch.cat([v_emb, c_emb], dim=1)
        fused = self._film(fused, self.condition_net(fused))
        fused = self.fusion_norm(fused)

        if return_fused:
            return fused

        reint_logit = self.reintervention_head(fused)

        if not self.complications:
            return (reint_logit,)

        comp_logits = self.complications_head(fused)
        return reint_logit, comp_logits
