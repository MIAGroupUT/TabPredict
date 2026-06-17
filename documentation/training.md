# Training Pipeline

## Overview

Training runs in two sequential stages. Each stage loads the best checkpoint from the previous one. Stage A warms up the tabular (clinical + AAA) branches while the visual backbone is fully frozen. Stage B unfreezes the BatchNorm layers of the I3ResNet image encoder (layer3/4) alongside the visual projection and CTA adapter, allowing scanner-domain adaptation.

The entry point is `scripts/train.py`, which loads a YAML config and calls `src.training.trainer.run_pipeline(cfg)`.

---

## Two-Stage Curriculum

### Stage A — Tabular Warm-Up

| Frozen | Trainable |
|---|---|
| `visual_branch` (Merlin ViT, all layers) | `clinical_branch` (MLP / TabM / classical wrapper) |
| `cta_adapter` | `reintervention_head` |
| `visual_proj` | `complications_head` (if present) |
| | `condition_net` (FiLM) |
| | `fusion_norm` |
| | `feature_gate` (**DL backbones only**) |

Goal: establish stable multimodal fusion on tabular data before the visual backbone is touched. The clinical features contain strong predictive signal, so this stage produces a meaningful baseline quickly.

**Classical ML backbones (LR / RF / GBM / XGBoost):** before Stage A begins, `fit_classical_backbone()` fits the sklearn model once on z-normalised clinical + raw AAA features from the training split. During Stage A (and B) only the `Linear(1→32)` projection inside `ClassicalBranchWrapper` trains — the sklearn model is fixed. The `feature_gate` is skipped entirely (classical models perform their own feature selection) and gate L1 regularisation is not applied.

### Stage B — Backbone Adaptation

Controlled by `freeze_clinical_in_stage_b` (YAML `training:` block, default `false`).

#### `freeze_clinical_in_stage_b: false` (default — joint fine-tuning)

| Frozen | Trainable |
|---|---|
| I3ResNet conv weights (layer1–4) | I3ResNet BatchNorm γ/β in layer3 and layer4 |
| | `cta_adapter` |
| | `visual_proj` |
| | `clinical_branch` (continues at reduced LR 5e-5) |
| | `condition_net` (re-initialised from scratch at stage start) |
| | `fusion_norm` |
| | `reintervention_head` |
| | `complications_head` (if present) |
| | `feature_gate` |

#### `freeze_clinical_in_stage_b: true` (visual-only adaptation)

| Frozen | Trainable |
|---|---|
| I3ResNet conv weights (layer1–4) | I3ResNet BatchNorm γ/β in layer3 and layer4 |
| `clinical_branch` | `cta_adapter` |
| `feature_gate` | `visual_proj` |
| | `condition_net` (re-initialised from scratch at stage start) |
| | `fusion_norm` |
| | `reintervention_head` |
| | `complications_head` (if present) |

Goal: recalibrate the Merlin backbone's feature statistics to the specific scanner types in this cohort (RNST, MST, UMCG, AUMC) using BatchNorm adaptation (~50 K params). `condition_net` is re-initialised at stage start so Stage A's clinical-only bias does not suppress the newly-active visual features. With `freeze_clinical_in_stage_b: true`, the clinical branch is locked after Stage A — Stage B becomes a pure visual adaptation pass.

---

## Loss Functions

### `ReinterventionLoss` — Primary Task

```
loss = mean( focal_BCE(logit, gt) )
```

Plain focal BCE averaged over the batch:

| Setting | Value | Purpose |
|---|---|---|
| `focal_gamma` | 3.0 | Focuses gradient on hard borderline examples in the 0.2–0.5 probability range |
| `label_smoothing` | 0.02 | Slightly softens hard 0/1 targets |
| `pos_weight` | 2.0 | Asymmetric gradient — false negatives penalised 2× false positives |

The forward returns `(loss_scalar, details_dict)` where `details_dict` contains `loss_total`, `loss_reint`.

### `MultiTaskReinterventionLoss` — With Complications

```
loss = loss_reint + comp_weight × mean(loss_comp_i)
```

- `loss_reint` is identical to the primary-task loss above
- Each `loss_comp_i` is a per-complication plain focal BCE
- Labels of **-1** are masked out — per-complication losses are only computed on rows with valid ground truth for that complication
- `comp_weight` is controlled by the YAML config and typically differs between stages (lower in Stage A, higher in Stage B once visual features are available)

### Gate L1 Regularisation

Added to the total loss in both variants:

```python
loss += gate_l1_weight × sigmoid(feature_gate).sum()
```

This pushes feature gate values toward zero for unimportant features, producing a sparse gate after training. Default 0.0001 — kept low because the gate now operates on ~42 pre-filtered features (missingness filter removes the 198 near-constant noise features). A higher L1 on the full 240-feature set caused the gate to zero out all clinical features including the informative ones.

---

## Optimizer

`build_optimizer(model, stage)` creates a single `AdamW` optimizer with distinct parameter groups and learning rates per group.

### Stage A parameter groups (5 or 6 with complications)

| Group name | Parameters | LR |
|---|---|---|
| `feature_gate` | Per-feature gate parameter | 1e-3 |
| `clinical` | `clinical_branch` weights | 3e-4 |
| `condition` | FiLM condition network | 2e-4 |
| `fusion_norm` | LayerNorm | 2e-4 |
| `reint_head` | Reintervention head | 2e-4 |
| `comp_head` | Complication heads (if present) | 2e-4 |

Visual branch parameters are not included (frozen → no gradients). Classical backbones omit `feature_gate` (no gradient path through sklearn).

### Stage B parameter groups

**`freeze_clinical_in_stage_b: false`** — 8 groups (9 with complications):

| Group name | Parameters | LR |
|---|---|---|
| `feature_gate` | Per-feature gate parameter | 3e-4 |
| `clinical` | `clinical_branch` weights | 5e-5 |
| `condition` | FiLM (re-initialised) | 5e-5 |
| `fusion_norm` | LayerNorm | 5e-5 |
| `cta_adapter` | CTA input adapter | 1e-5 |
| `visual_proj` | Visual projection layer | 1e-5 |
| `i3resnet_bn` | BatchNorm γ/β in I3ResNet layer3+4 | 1e-5 |
| `reint_head` | Reintervention head | 5e-5 |
| `comp_head` | Complication heads (if present) | 5e-5 |

**`freeze_clinical_in_stage_b: true`** — 6 groups (7 with complications):

| Group name | Parameters | LR |
|---|---|---|
| `condition` | FiLM (re-initialised) | 5e-5 |
| `fusion_norm` | LayerNorm | 5e-5 |
| `cta_adapter` | CTA input adapter | 1e-5 |
| `visual_proj` | Visual projection layer | 1e-5 |
| `i3resnet_bn` | BatchNorm γ/β in I3ResNet layer3+4 | 1e-5 |
| `reint_head` | Reintervention head | 5e-5 |
| `comp_head` | Complication heads (if present) | 5e-5 |

Only I3ResNet BN layers (layer3+4) are adapted — conv weights remain frozen. Layer3+4 BN adaptation recalibrates feature statistics across scanner protocols without touching learned conv filters.

### Learning Rate Schedule

Both stages use `CosineAnnealingWarmRestarts` with:
- Stage A: T₀ = 20 epochs
- Stage B: T₀ = 30 epochs

---

## Causal Modality Dropout

Applied during **both Stage A and Stage B training** to simulate real-world missingness and force all modalities to develop independent representations.

```
p(CTA missing)                          = p_cta           (0.10)
p(AAA missing | CTA present)            ≈ 1−(1−p_aaa_given_cta)×(1−p_aaa_given_clin)  (≈ 0.28 at default 0.15 each)
p(AAA missing | CTA absent)             = 0  (AAA is kept when CTA is dropped)
p(Clinical missing)                     = p_clinical      (0.15)
```

**AAA when CTA is absent:** AAA metrics may originate from clinical records or a prior measurement, so they are intentionally kept even when CTA is zeroed. When CTA is present, both `p_aaa_given_cta` and `p_aaa_given_clin` are checked independently, giving an effective combined dropout probability of ≈28% at the defaults.

**Clinical dropout:** Clinical data is always available in practice, but we drop it during training anyway. This prevents the visual branch from learning to defer to the tabular shortcut (which is already well-trained from Stage A).

At inference time, dropout is never applied. `maybe_apply_modality_dropout(is_training=False, ...)` is a strict no-op.

---

## Class Imbalance Handling

The training set has approximately 18% positive (reintervention) cases. The DataLoader uses a `WeightedRandomSampler` that draws positives and negatives with equal frequency (weight = 1/N_class per sample), producing roughly 50/50 batches. This eliminates the need to inflate `pos_weight` purely to counteract skew. `pos_weight = N_neg/N_pos ≈ 4.58` is still applied in the loss to encode the clinical preference: a missed positive (FN) is more costly than a false alarm (FP). The checkpoint is saved when the validation **F2-score** (β=2, weighting recall 4× over precision) improves, directly optimising for sensitivity rather than discrimination.

---

## Checkpoints

At the end of each epoch, if validation F2-score (recall_beta=2.0) improves, the trainer saves:

```python
{
    "state_dict":              model.state_dict(),
    "threshold":               best_threshold,         # F-beta optimal threshold
    "temperature":             temperature,            # post-hoc calibration scalar (set after all stages)
    "feature_names":           feature_names,          # ordered list of clinical feature names
    "clin_mean":               clin_mean,              # normalisation stats (array)
    "clin_std":                clin_std,
    "complications":           complications,          # list of complication names (or [])
    "clinical_backbone":       clinical_backbone,
    "tabm_cfg":                backbone_cfg,           # hyperparameters for the chosen backbone
    "config":                  cfg,                    # full YAML config dict
    # classical backbones only:
    "classical_baseline_bytes": bytes,                 # joblib-serialised sklearn pipeline
}
```

`load_checkpoint()` restores `classical_baseline_bytes` automatically into the `ClassicalBranchWrapper` when present, so `test.py` works with classical configs without any extra steps.

Checkpoints are written to `paths.checkpoint_dir` (see [Configuration](configuration.md)):
- `best_stage_a.pt` — best Stage A checkpoint
- `best_stage_b.pt` — best Stage B checkpoint (use this for inference)

---

## Early Stopping

Each stage has a configurable `patience` (number of epochs without F2-score improvement before stopping). Defaults: Stage A patience=12, Stage B patience=20.

---

## Train-Set Metrics

Every epoch the trainer evaluates both val and train sets by calling `validate()` twice (once with `split="val"`, once with `split="train"`). All metrics are logged to W&B under `val/...` and `train/...` prefixes:

```
[Stage A | Epoch 003]  train_loss=0.4123  val_loss=0.4891
[VAL]  loss=0.4891 | auc=0.61 | f1=0.54 | prec=0.52 | rec=0.57 | thresh=0.480
[TRAIN]  loss=0.4123 | auc=0.72 | f1=0.65 | prec=0.63 | rec=0.68 | thresh=0.475
```

W&B keys: `val/reint_auc`, `train/reint_auc`, `val/reint_f1`, `train/reint_f1`, etc.

Prediction tables (`eval/val_full_matrix_stage_{st}` and `eval/train_full_matrix_stage_{st}`) are logged every 3 epochs. At the end of each stage, ROC curves for both sets are uploaded as W&B images under `roc/val_stage_{st}` and `roc/train_stage_{st}`.

Note: train metrics reflect balanced batches from `WeightedRandomSampler` (50/50 class ratio), not the natural 18% positive rate, so the train AUC/F1 values are higher than they would be on the raw unbalanced training split.

---

## XAI During Training

Every `xai_every_n_epochs` epochs, the trainer runs `full_explain()` on `xai_samples` validation patients and logs a W&B table. For classical backbones, set `xai_every_n_epochs: 999` in the config (GradCAM requires an I3ResNet backbone; clinical gradient importance is less meaningful when the clinical encoder is non-differentiable). See [XAI](xai.md) and [W&B](wandb.md) for details.
