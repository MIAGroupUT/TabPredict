# Model Architecture

## Overview

`ReinterventionRiskModel` is a multi-modal, multi-task neural network. It fuses two input branches — a 3D visual branch and a unified tabular branch — into a shared 288-dimensional representation, which is then passed through task-specific output heads.

```
CTA volume (4-ch, 224×224×160)
    → cta_adapter (Conv3d 4→1)
    → Merlin I3ResNet (frozen, BN adapted in Stage B)
    → visual_proj (Linear 2048→256)           → 256-dim visual embedding ─────┐
                                                                              │
Clinical tabular (~42 features, z-normalised, filtered from 240 raw)          │
    → feature_gate (per-feature sigmoid) [DL backbones only]                  │
    → cat with AAA geometry (5 scalars, no gate)                              │
    → clinical_branch (MLP | TabM | LR | RF | GBM | XGBoost → 32-dim)         │
                                          → 32-dim tabular embedding ─────────┘
                                                        │
                                          cat → 288-dim fused representation
                                                        │
                                          FiLM (condition_net: γ/β, tanh×0.3)
                                          LayerNorm (fusion_norm)
                                                        │
                               ┌────────────────────────┤
                               │                        │
                   reintervention_head          complications_head
                   (logit)                      (logit per complication)
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

The 4-channel CTA input concatenates: CT intensity · lumen segmentation mask · ILT mask · lumen-to-wall distance map (see [Data](data.md) for runtime assembly; see [Preprocessing step 05](preprocessing.md#step-05--nnunet-segmentation--aaa-metrics-05_nnunet_labels_metrics_generationpy) for how the segmentation masks and distance map are produced from raw CT).

### Tabular Branch (Clinical + AAA Metrics)

| Component | Detail |
|---|---|
| `feature_gate` | Learnable `nn.Parameter` of shape `(clinical_dim,)`, passed through sigmoid at forward time. Gates each clinical feature independently. L1-regularised to encourage sparsity. |
| AAA append | The 5 AAA geometry scalars are concatenated to the gated clinical vector **after** gating, giving an input of `(clinical_dim + 5)` to the branch. They bypass the L1 gate since they are imaging-derived and always informative. |
| `clinical_branch` | Configurable backbone outputting **(B, 32)**. Input dim = `clinical_dim + aaa_dim`. Options: **MLP** (default), **TabM**, or a **classical ML wrapper** (LR / RF / GBM / XGBoost — see below). |
| Input | z-normalised clinical feature vector (~42 features, after missingness filter) + 5 AAA geometry scalars |

**Clinical feature filtering:** of the 240 raw RADAR `pre_op_features` fields, 198 (82%) are missing in ≥50% of training patients — these are imputed to the training mean, making them indistinguishable from an "average" patient and giving the gate only noise to work with. `filter_features_by_missingness()` (`src/data/loaders.py`) drops these at training time using the training split only (no leakage), reducing the input to ~42 genuinely informative features. The filtered `feature_names` list is saved in the checkpoint, so inference automatically uses the same subset.

The feature gate multiplies the normalised clinical input elementwise before appending the AAA scalars. Clinical features with gate values near zero are effectively suppressed while the geometry scalars remain unaffected. The L1 penalty (`gate_l1_weight`, default 0.0001) encourages selective sparsity over the ~42 filtered features.

**Why merge AAA scalars into the tabular branch?** A dedicated two-layer MLP for just 5 inputs (`Linear(5,32)→Linear(32,32)`) was heavily overparameterised (~1,200 params for 5 features) and gave the 5 scalars the same 32-dim "vote" in the fused representation as the 744-feature clinical branch. Appending them after gating shares the encoder capacity fairly and reduces the fusion dimension from 320 to 288, while keeping their signal intact (they bypass sparsity pressure via the gate).

#### The 5 AAA geometry scalars

```
volume_lumen_ml
volume_ilt_ml
volume_total_ml
max_lumen_to_wall_dist_mm
avg_lumen_to_wall_dist_mm
```

Derived from nnUNet segmentation masks by `preprocessing/case_matching_v1/05_nnUNet_labels_metrics_generation.py`. See [Preprocessing step 05](preprocessing.md#step-05--nnunet-segmentation--aaa-metrics-05_nnunet_labels_metrics_generationpy) for the exact formulas (voxel volumes, distance transform). Loaded at runtime from `metricsTr/{stem}.json`; missing keys default to 0.0.

#### TabM backbone (optional)

`clinical_backbone: tabm` switches the branch to **TabM** (Gorishniy et al., 2024 — "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"). TabM trains an ensemble of *k* MLPs with shared weights and per-member rank-1 adapters (BatchEnsemble). It returns `(B, k, 32)`; we reduce via ensemble mean to `(B, 32)`. TabM takes `clinical_dim + aaa_dim` features as input, so the AAA scalars are embedded together with the clinical features.

**Numerical feature embeddings (`PeriodicEmbeddings`, enabled by default):** before the MLP ensemble each scalar feature is projected into a `d_embedding`-dim sinusoidal representation — `[sin(f₁x), cos(f₁x), …]` — letting the network capture non-linear feature relationships. We use `lite=True` (shared frequencies across all features) rather than per-feature frequencies because with 749 features and only 377 training samples, per-feature frequencies introduce ~35 K extra params with too little data to learn them reliably. Shared frequencies cost just 48 params and are sufficient for our scale. Set `num_embeddings: none` to disable.

```yaml
model:
  clinical_backbone: tabm       # mlp (default) | tabm | lr | rf | gbm | xgboost
  backbone_cfg:
    k: 16                       # ensemble members
    n_blocks: 2                 # MLP depth per member
    d_block: 64                 # hidden width per member (halved from 128; ~4k params/positive)
    dropout: 0.3
    num_embeddings: periodic    # periodic (default) | none
    d_embedding: 8              # embedding dim per feature (halved from 16 to match d_block)
```

To benchmark without editing the YAML:
```bash
python scripts/train.py --config experiments/v1_no_complications.yaml \
    --set model.clinical_backbone=tabm
```

#### Classical ML backbones (LR / RF / GBM / XGBoost)

`clinical_backbone: lr | rf | gbm | xgboost` wraps a scikit-learn (or XGBoost) classifier inside `ClassicalBranchWrapper(nn.Module)` — a hybrid that combines a non-differentiable sklearn model with a learnable `Linear(1→32)` projection.

**How it works:**

1. Before Stage A, `fit_classical_backbone()` fits the sklearn model once on z-normalised clinical + raw AAA features from the training split (identical distribution to runtime inference).
2. During forward: the sklearn model's `predict_proba` output (a single scalar per patient) is projected to 32-dim via `Linear(1→32)`, giving a differentiable path into the FiLM fusion.
3. The visual branch, FiLM layer, and prediction head all train normally via backprop in both stages. The `feature_gate` is bypassed (classical models handle feature importance internally).
4. Checkpoints include `classical_baseline_bytes` — a joblib-serialised copy of the fitted sklearn pipeline — loaded automatically by `load_checkpoint()`.

This design means **CTA information is fully utilised** alongside the classical tabular model, unlike a pure tabular baseline. The classical backbone is a direct drop-in replacement for MLP/TabM — no code changes needed, just swap the YAML key.

```yaml
model:
  clinical_backbone: rf     # lr | rf | gbm | xgboost
  backbone_cfg:
    n_estimators: 300
    min_samples_leaf: 2
```

Dedicated configs: `experiments/v1_logistic_regression.yaml`, `v1_random_forest.yaml`, `v1_gradient_boosting.yaml`, `v1_xgboost.yaml`.

---

## FiLM Fusion

After concatenating the two branch embeddings (256 + 32 = **288-dim**), a FiLM (Feature-wise Linear Modulation) layer applies a learned affine transformation:

```
γ, β = condition_net(fused)          # MLP: Linear(288,256)→ReLU→Dropout(0.3)→Linear(256,576)
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

- `Linear(288, 256) → ReLU → Dropout(0.2) → Linear(256, 1)`

Predicted probability: `sigmoid(logit)`. Confidence: `2 × |prob − 0.5|` — zero when maximally uncertain (prob ≈ 0.5), one when certain (prob near 0 or 1). This is self-calibrating: if training produces well-separated probabilities, confidence is high automatically.

### `complications_head`

Only present when `complications` is not empty (multi-task variant). Produces one logit **per complication**:

- `Linear(288, 256) → ReLU → Dropout(0.2) → Linear(256, n_comp)`

The 7 complications supported (`COMPLICATIONS_FULL`):

```
endoleak_1a, endoleak_1b, endoleak_2, endoleak_3, endoleak_5,
migration, infection
```

---

## Forward Pass API

```python
from src.models.reintervention import ReinterventionRiskModel

# Primary-task model (default MLP clinical backbone)
# clinical_dim is derived from feature_names after missingness filtering (~42)
model = ReinterventionRiskModel(clinical_dim=42, aaa_dim=5, complications=[])
(logit,) = model(cta, clin, aaa)
# logit: (B, 1) — reintervention logit
# Internally: cat([clin * sigmoid(gate), aaa]) → clinical_branch → 32-dim

# Primary-task model with TabM clinical backbone
model = ReinterventionRiskModel(
    clinical_dim=42, aaa_dim=5, complications=[],
    clinical_backbone="tabm",
    tabm_cfg=dict(k=16, n_blocks=2, d_block=64, dropout=0.3,
                  num_embeddings="periodic", d_embedding=8),
)
(logit,) = model(cta, clin, aaa)

# Multi-task model
model = ReinterventionRiskModel(
    clinical_dim=42, aaa_dim=5,
    complications=["endoleak_1a", "endoleak_1b", "migration", ...],
)
logit, comp_logits = model(cta, clin, aaa)
# comp_logits: (B, n_comp)

# Return the fused 288-dim representation instead (useful for analysis)
fused = model(cta, clin, aaa, return_fused=True)   # (B, 288)
```

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Merlin I3ResNet backbone | Pretrained on 110k CT scans; strong 3D medical imaging priors without training from scratch |
| BN-only adaptation (layer3/4) | Recalibrates feature statistics per scanner type with ~50 K params; avoids catastrophic forgetting while correcting scanner-domain bias |
| FiLM clamped to ±0.3 | Allows meaningful cross-modal modulation (±30%) while staying stable during training |
| Per-feature sigmoid gate (L1-regularised) | Soft feature selection over ~42 clinical features (pre-filtered from 240 by missingness); interpretable — gate values can be inspected post-training |
| AAA scalars appended after gate | 5 imaging-derived geometry scalars bypass the sparsity gate (they are always informative) but share the clinical MLP/TabM encoder, avoiding the overparameterisation of a dedicated 5→32→32 branch |
| Plain focal BCE (no log-variance) | Simpler and equally effective; confidence is derived from `2×\|prob−0.5\|` which is self-calibrating without post-hoc tuning |
| Configurable clinical backbone | MLP is the proven default; TabM (parameter-efficient ensemble) or classical ML (LR / RF / GBM / XGBoost) can be swapped in via a single YAML key. Classical backbones use `ClassicalBranchWrapper` — a fitted sklearn model + learnable projection — so the visual branch still trains and CTA information is preserved |
| F-beta (β=2) threshold | Weights recall 4× over precision — missing a reintervention is clinically worse than a false alarm |
| Causal modality dropout | Encodes clinical constraint (AAA metrics cannot exist without CTA); forces all branches to develop independent representations |

---

## Helper Functions

```python
from src.models.reintervention import get_model, load_checkpoint, disable_checkpointing, is_classical_backbone

# Unwrap DataParallel safely
model = get_model(model_or_wrapped)

# Load a checkpoint; strips "module." prefix from DataParallel checkpoints.
# Also restores classical_baseline_bytes (fitted sklearn model) when present.
ckpt = load_checkpoint(model, path, device="cuda", strict=True)
threshold   = ckpt["threshold"]
feat_names  = ckpt["feature_names"]

# Check backbone type
if is_classical_backbone(model):
    print("Using classical ML tabular backbone")

# Disable gradient checkpointing (required before XAI gradient computation)
disable_checkpointing(model)
```
