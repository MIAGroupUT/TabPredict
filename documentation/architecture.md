# Model Architecture

## Overview

`ReinterventionRiskModel` is a multi-modal, multi-task neural network. It fuses three independent input branches — a 3D visual branch, a clinical tabular branch, and an AAA geometry branch — into a shared 320-dimensional representation, which is then passed through task-specific output heads.

```
CTA volume (4-ch, 224×224×160)
    → cta_adapter (Conv3d 4→1)
    → Merlin I3ResNet (frozen, BN adapted in Stage B)
    → visual_proj (Linear 2048→256)           → 256-dim visual embedding ─────┐
                                                                              │
Clinical tabular (~744 features, z-normalised)                                │
    → feature_gate (per-feature sigmoid)                                      │
    → clinical_branch (MLP → 32-dim)          → 32-dim clinical embedding ────┤
                                                                              │
AAA geometry (5 scalars)                                                      │
    → aaa_metrics_branch (MLP → 32-dim)       → 32-dim AAA embedding ─────────┘
                                                        │
                                          cat → 320-dim fused representation
                                                        │
                                          FiLM (condition_net: γ/β, tanh×0.3)
                                          LayerNorm (fusion_norm)
                                                        │
                               ┌────────────────────────┤
                               │                        │
                   reintervention_head          complications_head
                   (logit, log-variance)        (logit, log-variance per comp)
                                                [only if complications ≠ []]
```

---

## Input Branches

### Visual Branch

| Component | Detail |
|---|---|
| `cta_adapter` | `Conv3d(4→8→1)` — projects the 4-channel CTA tensor to 1 channel before Merlin |
| Backbone | Merlin I3ResNet pretrained on 110k CT scans; produces a **(B, 2048)** embedding |
| Adaptation | Stage B unfreezes BatchNorm γ/β in I3ResNet `layer3` and `layer4` (~50 K params); recalibrates feature statistics for scanner-type variation |
| `visual_proj` | `Linear(2048, 256)` — projects to shared embedding dimension |

The 4-channel CTA input concatenates: CT intensity · lumen segmentation mask · ILT segmentation mask · distance map (see [Data](data.md) for how this is assembled).

### Clinical Branch

| Component | Detail |
|---|---|
| `feature_gate` | Learnable `nn.Parameter` of shape `(clinical_dim,)`, passed through sigmoid at forward time. Gates each clinical feature independently. L1-regularised to encourage sparsity. |
| `clinical_branch` | Linear probe: `Linear(clinical_dim, 32)` — single layer, no hidden MLP. With ~37 training positives a MLP memorises; a linear probe cannot. |
| Input | z-normalised feature vector (~744 features from RADAR); see [Data](data.md) |

The feature gate multiplies the normalised input elementwise before passing it to the linear probe. Features with gate values near zero are effectively suppressed. The L1 penalty (`gate_l1_weight`, default 0.005) forces the gate to keep only ~20–30 features, preventing clinical branch overfitting with the ~37 training positives available.

### AAA Metrics Branch

| Component | Detail |
|---|---|
| `aaa_metrics_branch` | Two-layer MLP: `Linear(5, 32) → ReLU → Linear(32, 32) → ReLU` |
| Input | 5 geometry scalars derived from nnUNet segmentation (see [Data](data.md)) |

The 5 metrics are:

```
volume_lumen_ml
volume_ilt_ml
volume_total_ml
max_lumen_to_wall_dist_mm
avg_lumen_to_wall_dist_mm
```

---

## FiLM Fusion

After concatenating the three branch embeddings (256 + 32 + 32 = **320-dim**), a FiLM (Feature-wise Linear Modulation) layer applies a learned affine transformation:

```
γ, β = condition_net(fused)          # MLP: Linear(320,256)→ReLU→Dropout(0.3)→Linear(256,640)
γ = tanh(γ) × 0.3                    # clamp to (-0.3, +0.3)
β = tanh(β) × 0.3
fused = (1 + γ) × fused + β          # modulate
fused = fusion_norm(fused)           # LayerNorm
```

The ±0.3 clamp on γ/β allows meaningful cross-modal modulation (±30%) while staying stable during training.

---

## Output Heads

Each head produces a single **logit** per task (no log-variance). Confidence is derived from the predicted probability, not from a learned uncertainty term.

### `reintervention_head`

- `Linear(320, 256) → ReLU → Dropout(0.2) → Linear(256, 1)`

Predicted probability: `sigmoid(logit)`. Confidence: `2 × |prob − 0.5|` — zero when maximally uncertain (prob ≈ 0.5), one when certain (prob near 0 or 1). This is self-calibrating: if training produces well-separated probabilities, confidence is high automatically.

### `complications_head`

Only present when `complications` is not empty (multi-task variant). Produces one logit **per complication**:

- `Linear(320, 256) → ReLU → Dropout(0.2) → Linear(256, n_comp)`

The 7 complications supported (`COMPLICATIONS_FULL`):

```
endoleak_1a, endoleak_1b, endoleak_2, endoleak_3, endoleak_5,
migration, infection
```

---

## Forward Pass API

```python
from src.models.reintervention import ReinterventionRiskModel

# Primary-task model
model = ReinterventionRiskModel(clinical_dim=744, aaa_dim=5, complications=[])
(logit,) = model(cta, clin, aaa)
# logit: (B, 1) — reintervention logit

# Multi-task model
model = ReinterventionRiskModel(
    clinical_dim=744, aaa_dim=5,
    complications=["endoleak_1a", "endoleak_1b", "migration", ...],
)
logit, comp_logits = model(cta, clin, aaa)
# comp_logits: (B, n_comp)

# Return the fused 320-dim representation instead (useful for analysis)
fused = model(cta, clin, aaa, return_fused=True)   # (B, 320)
```

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Merlin I3ResNet backbone | Pretrained on 110k CT scans; strong 3D medical imaging priors without training from scratch |
| BN-only adaptation (layer3/4) | Recalibrates feature statistics per scanner type with ~50 K params; avoids catastrophic forgetting while correcting scanner-domain bias |
| FiLM clamped to ±0.3 | Allows meaningful cross-modal modulation (±30%) while staying stable during training |
| Per-feature sigmoid gate (L1-regularised) | Soft feature selection over ~744 clinical features; interpretable — gate values can be inspected post-training |
| Heteroscedastic log-variance, clamped [-4, 2] | Lets the model express calibrated confidence; quadratic penalty prevents degenerate high-uncertainty predictions |
| F-beta (β=2) threshold | Weights recall 4× over precision — missing a reintervention is clinically worse than a false alarm |
| Causal modality dropout | Encodes clinical constraint (AAA metrics cannot exist without CTA); forces all branches to develop independent representations |

---

## Helper Functions

```python
from src.models.reintervention import get_model, load_checkpoint, disable_checkpointing

# Unwrap DataParallel safely
model = get_model(model_or_wrapped)

# Load a checkpoint; strips "module." prefix from DataParallel checkpoints
ckpt = load_checkpoint(model, path, device="cuda", strict=True)
threshold   = ckpt["threshold"]
feat_names  = ckpt["feature_names"]

# Disable gradient checkpointing (required before XAI gradient computation)
disable_checkpointing(model)
```
