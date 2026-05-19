# Training Pipeline

## Overview

Training runs in two sequential stages. Each stage loads the best checkpoint from the previous one. Stage A warms up the tabular (clinical + AAA) branches while the visual backbone is frozen. Stage B unfreezes the LoRA adapters and adapts the backbone to the task.

The entry point is `scripts/train.py`, which loads a YAML config and calls `src.training.trainer.run_pipeline(cfg)`.

---

## Two-Stage Curriculum

### Stage A — Tabular Warm-Up

| Frozen | Trainable |
|---|---|
| `visual_branch` (Merlin ViT, all layers) | `clinical_branch` (MLP) |
| `cta_adapter` | `aaa_metrics_branch` (MLP) |
| `visual_proj` | `reintervention_head` |
| | `complications_head` (if present) |
| | `condition_net` (FiLM) |
| | `fusion_norm` |
| | `feature_gate` |

Goal: establish stable multimodal fusion on tabular data before the visual backbone is touched. The clinical features contain strong predictive signal, so this stage produces a meaningful baseline quickly.

### Stage B — Backbone Adaptation

| Frozen | Trainable |
|---|---|
| `clinical_branch` | `cta_adapter` |
| `aaa_metrics_branch` | `visual_proj` |
| | LoRA adapters on Merlin query/value projections |
| | `condition_net` (re-initialised from scratch at stage start) |
| | `fusion_norm` |
| | `reintervention_head` |
| | `complications_head` (if present) |
| | `feature_gate` |

Goal: adapt the Merlin backbone to this specific task via LoRA, with the tabular branches held fixed to prevent the visual branch from collapsing onto the tabular shortcut. `condition_net` is re-initialised at stage start to remove Stage A's clinical bias from the FiLM modulation.

> **Why freeze the clinical branch in Stage B?** Without this, the model can always improve its training loss by routing more signal through the (already-warm) clinical branch rather than learning from CTA features. Freezing forces the visual LoRA adapters to contribute independently.

---

## Loss Functions

### `ReinterventionLoss` — Primary Task

```
loss = exp(−logvar) × focal_BCE(logit, gt) + 0.5 × logvar + 0.2 × logvar²
```

Component by component:

| Term | Purpose |
|---|---|
| `exp(−logvar) × focal_BCE` | Down-weights loss when the model is uncertain; focal weighting (γ=2) reduces the contribution of easy negatives |
| `+ 0.5 × logvar` | Linear uncertainty penalty — prevents the model from suppressing all gradients by predicting maximum uncertainty |
| `+ 0.2 × logvar²` | Quadratic penalty — stronger pull away from the upper logvar clamp (logvar=2 would otherwise suppress the BCE gradient entirely) |

Additional settings:
- **Label smoothing** ε = 0.02 — slightly softens the hard 0/1 targets
- **pos_weight** = 3.0 — extra penalty for false negatives; combined with a balanced sampler, this gives re-intervention cases extra emphasis
- **logvar clamp** [-4, 2] applied inside the forward pass before the loss is computed

The forward returns `(loss_scalar, details_dict)` where `details_dict` contains `loss_total`, `loss_reint`, `mean_uncertainty`.

### `MultiTaskReinterventionLoss` — With Complications

```
loss = loss_reint + comp_weight × mean(loss_comp_i)
```

- `loss_reint` is identical to the primary-task loss above
- Each `loss_comp_i` is a per-complication focal BCE with heteroscedastic weighting
- Labels of **-1** are masked out — per-complication losses are only computed on rows with valid ground truth for that complication
- `comp_weight` is controlled by the YAML config and typically differs between stages (lower in Stage A, higher in Stage B once visual features are available)

### Gate L1 Regularisation

Added to the total loss in both variants:

```python
loss += gate_l1_weight × sigmoid(feature_gate).sum()
```

This pushes feature gate values toward zero for unimportant features, producing a sparse gate after training. The weight is kept small (default 0.0001) to avoid over-suppressing features on this dataset size.

---

## Optimizer

`build_optimizer(model, stage)` creates a single `AdamW` optimizer with distinct parameter groups and learning rates per group.

### Stage A parameter groups (6 or 7 with complications)

| Group name | Parameters | LR |
|---|---|---|
| `clinical` | `clinical_branch` weights | 5e-4 |
| `aaa` | `aaa_metrics_branch` weights | 5e-4 |
| `condition_net` | FiLM condition network | 1e-4 |
| `reintervention_head` | Reintervention head | 1e-4 |
| `feature_gate` | Per-feature gate parameter | 1e-3 |
| `complications_head` | Complication heads (if present) | 1e-4 |

Visual branch parameters are not included (frozen → no gradients).

### Stage B parameter groups (9 or 10 with complications)

| Group name | Parameters | LR |
|---|---|---|
| `visual` | Merlin ViT base weights | 5e-6 |
| `lora` | LoRA adapter weights | 5e-6 |
| `cta_adapter` | CTA input adapter | 1e-4 |
| `visual_proj` | Visual projection layer | 1e-4 |
| `condition_net` | FiLM (re-initialised) | 1e-4 |
| `reintervention_head` | Reintervention head | 1e-4 |
| `feature_gate` | Per-feature gate | 1e-3 |
| `fusion_norm` | LayerNorm | 1e-4 |
| `complications_head` | Complication heads (if present) | 1e-4 |

Clinical and AAA branches are not included in Stage B (frozen).

### Learning Rate Schedule

Both stages use `CosineAnnealingWarmRestarts` with:
- Stage A: T₀ = 20 epochs
- Stage B: T₀ = 30 epochs

---

## Causal Modality Dropout

Applied only during **Stage B training** to simulate real-world missingness and force all modalities to develop independent representations.

```
p(CTA missing)                = p_cta           (default 0.10)
p(AAA missing | CTA dropped)  = p_aaa_given_cta  (default 0.15)
p(AAA missing | CTA present)  = p_aaa_given_clin (default 0.15)
p(Clinical missing)           = p_clinical      (default 0.20)
```

**Causal constraint:** AAA metrics are derived from CTA segmentation, so they cannot logically be available when the CTA is absent. The dropout logic encodes this: when CTA is zeroed out for a sample, the AAA tensor for that sample is also zeroed out with additional probability.

**Clinical dropout:** Clinical data is always available in practice, but we drop it during training anyway. This prevents the visual branch from learning to defer to the tabular shortcut (which is already well-trained from Stage A).

At inference time, dropout is never applied. `maybe_apply_modality_dropout(is_training=False, ...)` is a strict no-op.

---

## Balanced Sampler

The training DataLoader uses a weighted sampler so that each batch sees approximately equal numbers of re-intervention positive and negative cases. Combined with `pos_weight=3.0` in the loss, this gives the model two layers of correction for the class imbalance typical in EVAR cohorts.

---

## Checkpoints

At the end of each epoch, if validation AUC improves, the trainer saves:

```python
{
    "state_dict":     model.state_dict(),
    "threshold":      best_threshold,      # F-beta optimal threshold
    "feature_names":  feature_names,       # ordered list of clinical feature names
    "clin_mean":      clin_mean,           # normalisation stats (array)
    "clin_std":       clin_std,
    "complications":  complications,       # list of complication names (or [])
    "config":         cfg,                 # full YAML config dict
}
```

Checkpoints are written to `paths.checkpoint_dir` (see [Configuration](configuration.md)):
- `best_stage_a.pt` — best Stage A checkpoint
- `best_stage_b.pt` — best Stage B checkpoint (use this for inference)

---

## Early Stopping

Each stage has a configurable `patience` (number of epochs without AUC improvement before stopping). Defaults: Stage A patience=12, Stage B patience=20.

---

## XAI During Training

Every `xai_every_n_epochs` epochs, the trainer runs `full_explain()` on `xai_samples` validation patients and logs a W&B table. See [XAI](xai.md) and [W&B](wandb.md) for details.
