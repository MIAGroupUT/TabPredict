"""
Training stage management: seed, freeze/unfreeze, optimizer construction,
modality dropout, and parameter accounting.

Stage A — Frozen Backbone (Fusion Warm-Up)
    Merlin/LoRA fully frozen.  All other branches train at moderate LR.
    Goal: stable multimodal fusion before adapting the visual backbone.

Stage B — LoRA Fine-Tuning (Backbone Adaptation)
    LoRA adapters unfrozen at very low LR.  condition_net re-initialised
    so Stage A clinical bias does not suppress visual features.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from src.models.reintervention import get_model


# ── reproducibility ───────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ── modality dropout (causal augmentation) ───────────────────────────────────

def causal_modality_dropout(
    cta: torch.Tensor,
    clin: torch.Tensor,
    aaa: torch.Tensor,
    p_cta: float = 0.1,
    p_aaa_given_cta: float = 0.2,
    p_aaa_given_clin: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate realistic missingness patterns.
    - CTA goes missing independently with p_cta.
    - AAA metrics (derived from CTA) go missing when CTA is missing OR with
      additional probability reflecting poor-quality segmentation.
    """
    bsz = cta.size(0)
    cta_d, clin_d, aaa_d = cta.clone(), clin.clone(), aaa.clone()
    cta_present = torch.rand(bsz, device=cta.device) > p_cta

    for b in range(bsz):
        if not cta_present[b]:
            cta_d[b] = 0
            continue  # AAA may still be derivable from clinical
        if random.random() < p_aaa_given_cta or random.random() < p_aaa_given_clin:
            aaa_d[b] = 0

    return cta_d, clin_d, aaa_d


def maybe_apply_modality_dropout(
    is_training: bool,
    cta: torch.Tensor,
    clin: torch.Tensor,
    aaa: torch.Tensor,
    p_cta: float,
    p_aaa_given_cta: float,
    p_aaa_given_clin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not is_training:
        return cta, clin, aaa
    return causal_modality_dropout(cta, clin, aaa, p_cta, p_aaa_given_cta, p_aaa_given_clin)


# ── stage setup ───────────────────────────────────────────────────────────────

def set_stage_a(model: nn.Module) -> None:
    """
    Freeze Merlin/LoRA backbone entirely.  Train everything else:
    cta_adapter, visual_proj, feature_gate, clinical/AAA branches,
    condition_net, fusion_norm, reintervention_head, complications_head
    (if present).
    """
    model = get_model(model)
    for p in model.parameters():
        p.requires_grad = False

    for p in model.cta_adapter.parameters():
        p.requires_grad = True
    model.visual_proj.requires_grad_(True)
    model.feature_gate.requires_grad = True
    for p in model.clinical_branch.parameters():
        p.requires_grad = True
    for p in model.aaa_metrics_branch.parameters():
        p.requires_grad = True
    for p in model.condition_net.parameters():
        p.requires_grad = True
    for p in model.fusion_norm.parameters():
        p.requires_grad = True
    for p in model.reintervention_head.parameters():
        p.requires_grad = True
    if hasattr(model, "complications_head"):
        for p in model.complications_head.parameters():
            p.requires_grad = True


def set_stage_b(model: nn.Module) -> None:
    """
    Add LoRA adapter training on top of Stage A.
    Re-initialise condition_net so it relearns with visual LoRA features.
    """
    model = get_model(model)
    for p in model.parameters():
        p.requires_grad = False

    # Reset FiLM conditioning so Stage A clinical bias doesn't dominate
    for m in model.condition_net.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    # Same branches as Stage A
    for p in model.cta_adapter.parameters():
        p.requires_grad = True
    model.visual_proj.requires_grad_(True)
    model.feature_gate.requires_grad = True
    for p in model.clinical_branch.parameters():
        p.requires_grad = True
    for p in model.aaa_metrics_branch.parameters():
        p.requires_grad = True
    for p in model.condition_net.parameters():
        p.requires_grad = True
    for p in model.fusion_norm.parameters():
        p.requires_grad = True
    for p in model.reintervention_head.parameters():
        p.requires_grad = True
    if hasattr(model, "complications_head"):
        for p in model.complications_head.parameters():
            p.requires_grad = True

    # LoRA adapters in Merlin at very low LR
    for name, p in model.visual_branch.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True


# ── optimizer construction ────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, stage: str) -> torch.optim.Optimizer:
    """Stage-specific learning rates (AdamW with weight decay)."""
    model = get_model(model)

    if stage == "A":
        param_groups = [
            {"params": model.cta_adapter.parameters(),        "lr": 2e-4, "name": "cta_adapter"},
            {"params": model.visual_proj.parameters(),         "lr": 2e-4, "name": "visual_proj"},
            {"params": [model.feature_gate],                   "lr": 1e-3, "name": "feature_gate"},
            {"params": model.clinical_branch.parameters(),     "lr": 3e-4, "name": "clinical"},
            {"params": model.aaa_metrics_branch.parameters(),  "lr": 3e-4, "name": "aaa"},
            {"params": model.condition_net.parameters(),       "lr": 2e-4, "name": "condition"},
            {"params": model.fusion_norm.parameters(),         "lr": 2e-4, "name": "fusion_norm"},
            {"params": model.reintervention_head.parameters(), "lr": 2e-4, "name": "reint_head"},
        ]
        if hasattr(model, "complications_head"):
            param_groups.append(
                {"params": model.complications_head.parameters(), "lr": 2e-4, "name": "comp_head"}
            )

    elif stage == "B":
        lora_params = [p for n, p in model.visual_branch.named_parameters()
                       if "lora" in n.lower()]
        param_groups = [
            {"params": [model.feature_gate],                   "lr": 3e-4, "name": "feature_gate"},
            {"params": model.clinical_branch.parameters(),     "lr": 5e-5, "name": "clinical"},
            {"params": model.aaa_metrics_branch.parameters(),  "lr": 5e-5, "name": "aaa"},
            {"params": model.condition_net.parameters(),       "lr": 5e-5, "name": "condition"},
            {"params": model.fusion_norm.parameters(),         "lr": 5e-5, "name": "fusion_norm"},
            {"params": model.cta_adapter.parameters(),         "lr": 1e-5, "name": "cta_adapter"},
            {"params": lora_params,                            "lr": 5e-6, "name": "lora"},
            {"params": model.visual_proj.parameters(),         "lr": 1e-5, "name": "visual_proj"},
            {"params": model.reintervention_head.parameters(), "lr": 5e-5, "name": "reint_head"},
        ]
        if hasattr(model, "complications_head"):
            param_groups.append(
                {"params": model.complications_head.parameters(), "lr": 5e-5, "name": "comp_head"}
            )
    else:
        raise ValueError(f"Unknown stage: {stage!r}")

    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


# ── parameter accounting ──────────────────────────────────────────────────────

def count_trainable(obj: nn.Module | nn.Parameter) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    if isinstance(obj, nn.Parameter):
        return (obj.numel() if obj.requires_grad else 0), obj.numel()
    total = trainable = 0
    for p in obj.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def print_model_param_overview(model: nn.Module) -> None:
    model = get_model(model)
    sections = [
        ("clinical_branch",     model.clinical_branch),
        ("feature_gate",        model.feature_gate),
        ("aaa_metrics_branch",  model.aaa_metrics_branch),
        ("visual_branch (LoRA)", model.visual_branch),
        ("visual_proj",         model.visual_proj),
        ("condition_net",       model.condition_net),
        ("fusion_norm",         model.fusion_norm),
        ("reintervention_head", model.reintervention_head),
    ]
    if hasattr(model, "complications_head"):
        sections.append(("complications_head", model.complications_head))
    for name, obj in sections:
        tr, tot = count_trainable(obj)
        pct = 100 * tr / max(tot, 1)
        print(f"  {name:<25}  {tr:>10,} / {tot:>10,}  ({pct:.1f}%)")
