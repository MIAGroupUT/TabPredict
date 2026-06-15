# Tabular Backbone Models

Technical reference for all `clinical_branch` options in `ReinterventionRiskModel`. Every backbone satisfies the same contract: `(B, clinical_dim + aaa_dim) → (B, 32)`.

---

## Shared Data Pipeline

All backbones receive the same pre-processed tabular tensor from the DataLoader:

```
Raw RADAR features (240 columns)
    → missingness filter (<50% missing → ~42 features kept)
    → NormalizeClinicald transform
          z-score with training-set mean/std
          NaN → z-scored training-set median (clin_fill)
    → cat(clinical [~42], AAA scalars [5])  ← AAA bypass the gate
    → clinical_branch input  (B, ~47)
```

The **L1 feature gate** sits between the transform and the backbone for differentiable backbones only. It is a per-clinical-feature `sigmoid` with L1 regularisation (`gate_l1_weight=0.0001`) that pushes irrelevant features toward zero. AAA scalars are concatenated *after* the gate so they are never suppressed.

---

## Backbone Comparison Table

| Backbone | Class | Differentiable | Feature Gate | Imputation | Projection | Fit Timing | Params (approx.) |
|----------|-------|:--------------:|:------------:|------------|------------|------------|------------------|
| `mlp` | `MLP` | ✓ | ✓ | DataLoader (NaN→z-median) | Built-in (→32) | Stage A + B | ~clinical_dim×128 + 128×32 |
| `tabm` | `TabMEmbed` | ✓ | ✓ | DataLoader (NaN→z-median) | Built-in (→32) | Stage A + B | k×mlp_params + PeriodicEmbed |
| `saint` | `SAINTEmbed` | ✓ | ✓ | DataLoader (NaN→z-median) | Built-in (→32) | Stage A + B | ~66k (col) / ~850k (colrow) |
| `tabpfn` | `TabPFNBranchWrapper` | Partial* | ✓ | DataLoader (NaN→z-median) | `Linear(n_est×2→32)` | Phase 1 pre-A + Phase 2 A+B | 288 (projection) + last-N blocks |
| `lr` | `ClassicalBranchWrapper` | ✗ | ✗ | Median (TabularBaseline) | `Linear(1→32)` | Before Stage A | ~32 (projection) |
| `rf` | `ClassicalBranchWrapper` | ✗ | ✗ | Median (TabularBaseline) | `Linear(1→32)`† | Before Stage A | ~32† (projection) |
| `gbm` | `ClassicalBranchWrapper` | ✗ | ✗ | Median (TabularBaseline) | `Linear(1→32)`† | Before Stage A | ~32† (projection) |
| `xgboost` | `ClassicalBranchWrapper` | ✗ | ✗ | **None** (native NaN splits) | `Linear(1→32)`† | Before Stage A | ~32† (projection) |

†Default (`use_tree_output: false`). Set `use_tree_output: true` in `backbone_cfg` to use the richer per-tree representation: rf/gbm → `Linear(n_est→32)` (~9.6k params), xgboost → `Linear(n_feat→32)` (~1.5k params). Richer projections require larger datasets (>5k samples) to outperform the scalar default on RADAR.

*Last `n_trainable_layers` (default 2) transformer blocks trainable; remainder frozen.

---

## Projection Layer — Why and When

**Differentiable backbones (mlp / tabm / saint):** The 32-dim output is a design choice baked into the backbone architecture itself. The final `nn.Linear` of the backbone *is* the projection — no separate layer is needed.

**Classical backbones (lr / rf / gbm / xgboost):** The sklearn/XGBoost model is non-differentiable; its output is extracted as a structured representation and projected into the 32-dim fusion space. The projection is the *only* trainable component during Stage A + B.

Each model exposes a different input to the projection, controlled by the `use_tree_output` flag in `backbone_cfg`:

**`use_tree_output: false` (default — recommended for RADAR)**

All tree-based backends collapse to a single calibrated probability. The projection is `Linear(1→32)` with only 64 trainable parameters — optimal for small datasets (~700 training samples) where a wider projection overfits severely.

| Model | Representation | Shape | Projection |
|-------|---------------|-------|------------|
| `lr` | Scalar positive-class probability | `(B, 1)` | `Linear(1→32)` |
| `rf` | Scalar positive-class probability | `(B, 1)` | `Linear(1→32)` |
| `gbm` | Scalar positive-class probability | `(B, 1)` | `Linear(1→32)` |
| `xgboost` | Scalar positive-class probability | `(B, 1)` | `Linear(1→32)` |

**`use_tree_output: true` (richer projections — for larger datasets)**

| Model | Representation | Shape | Projection | Params |
|-------|---------------|-------|------------|--------|
| `lr` | Scalar probability (unchanged) | `(B, 1)` | `Linear(1→32)` | ~64 |
| `rf` | Per-tree positive-class probability | `(B, n_est)` | `Linear(n_est→32)` | ~9.6k |
| `gbm` | Per-tree leaf index / max_per_tree | `(B, n_est)` | `Linear(n_est→32)` | ~9.6k |
| `xgboost` | TreeSHAP per input feature (bias dropped) | `(B, n_feat)` | `Linear(n_feat→32)` | ~1.5k |

The richer representations preserve ensemble diversity (rf/gbm) and feature attribution signal (xgboost). However, the projection is the *only* trainable component (the sklearn model is frozen), so a wide projection on a small dataset produces a 14:1 parameter-to-sample ratio for rf/gbm — leading to overfitting and ~10% lower test AUC on RADAR.

The GBM leaf scale (`max_per_tree`) is computed from the training set during `fit()` and stored on the `TabularBaseline` object so `joblib.dump` serialises it into the checkpoint alongside the sklearn model.

**TabPFN:** Produces `(n_estimators, B, n_classes)` raw logits — one pair of class logits per estimator, before softmax. Flattened to `(B, n_estimators × 2)` and projected with `Linear(n_est×2→32)`. The richer input vs. a scalar probability preserves ensemble diversity: each estimator used a different random preprocessing seed, so differences in logit magnitude and sign carry genuine signal.

---

## Imputation — Model-by-Model

| Backbone | Where | How |
|----------|-------|-----|
| `mlp`, `tabm`, `saint`, `tabpfn` | `NormalizeClinicald` (DataLoader) | NaN → z-scored training-set median (`clin_fill = (median - mean) / std`) |
| `lr`, `rf`, `gbm` | `TabularBaseline._impute()` | NaN → column median computed on the (z-scored) training features |
| `xgboost` | **No imputation** | `TabularBaseline._impute()` is a deliberate no-op. XGBoost learns the optimal default branch direction for NaN splits during tree construction — overwriting NaN with a fixed value would discard this learned signal. |

> **Note:** `NormalizeClinicald` (DataLoader) fills NaN with `clin_fill` before the tensor reaches any backbone. The `TabularBaseline._impute()` therefore acts as a safety net for any raw data paths (e.g. standalone `TabularBaseline` usage outside the DataLoader). In normal multi-modal training all NaN are already filled by the DataLoader step.

---

## Execution Flow per Backbone Type

### Differentiable (mlp / tabm / saint)

```
fit_classical_backbone()    →  not called
Stage A                     →  full gradient through backbone + gate + fusion
Stage B                     →  same, plus visual pathway unfreezes
```


### TabPFN (tabpfn)

```
fit_classical_backbone()    →  Phase 1: FinetunedTabPFNClassifier.fit()
                               meta-learning fine-tune (full weights, AdamW lr=1e-5,
                               ~20 epochs, early-stop patience=5)
                               Phase 2 setup: differentiable_input=True,
                               fit_with_differentiable_input() initialises
                               InferenceEngineCachePreprocessing(inference_mode=False)
                               Freeze all TabPFN params;
                               unfreeze last n_trainable_layers encoder/decoder blocks
Stage A                     →  last N blocks + projection + gate + fusion train
Stage B                     →  same; visual pathway also unfreezes
checkpoint                  →  FinetunedTabPFNClassifier serialised to
                               classical_baseline_bytes (joblib)
                               restore_from_bytes() re-establishes differentiable mode
```


### Classical / Non-differentiable (lr / rf / gbm / xgboost)

```
fit_classical_backbone()    →  sklearn .fit() on z-scored train features
                               (XGBoost: NaN kept; others: NaN → median)
Stage A                     →  backbone frozen; only projection + fusion train
Stage B                     →  same; visual pathway also unfreezes
checkpoint                  →  sklearn model serialised to classical_baseline_bytes
```

---

## Hyperparameters

### MLP (`mlp`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 128 | Width of both hidden layers |
| dropout | 0.1 | Dropout rate |
| out_dim | 32 | Output embedding size |

### TabM (`tabm`)

Configured via `tabm_cfg` in the experiment YAML. Key params:

| Parameter | Default | Description |
|-----------|---------|-------------|
| k | 32 | Number of MLP ensemble members |
| d_embedding | 16 | PeriodicEmbeddings output dim |
| out_dim | 32 | Output embedding size |

### SAINT (`saint`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| style | `col` | `col` = column attention only (~66k params); `colrow` = column + intersample (~850k) |
| dim | 32 | Embedding dim |
| depth | 3 | Number of transformer blocks |
| heads | 4 | Attention heads |
| out_dim | 32 | Output embedding size |

### Classical (lr / rf / gbm / xgboost)

| Model | Key params | Notes |
|-------|-----------|-------|
| `lr` | `C=1.0`, `max_iter=2000`, `class_weight=balanced` | LBFGS solver |
| `rf` | `n_estimators=300`, `class_weight=balanced` | CPU parallel |
| `gbm` | `n_estimators=300`, `lr=0.05`, `max_depth=3`, `subsample=0.8` | sklearn GBM |
| `xgboost` | `n_estimators=300`, `lr=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8` | `scale_pos_weight` auto-set from class counts |

All classical models sit inside a `StandardScaler` sklearn pipeline (except XGBoost which is scale-invariant due to tree splits).

### TabPFN (`tabpfn`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 20 | Phase 1 meta-learning epochs (early-stop may trigger first) |
| learning_rate | 1e-5 | AdamW lr for phase 1 fine-tuning |
| early_stopping_patience | 5 | Patience for phase 1 early stopping |
| n_estimators_final_inference | 4 | Ensemble size at inference → projection input = 4×2=8 |
| n_trainable_layers | 2 | Last N encoder/decoder blocks unfrozen in phase 2 |
| random_state | 42 | Reproducibility seed |

Pretrained weights downloaded automatically from HuggingFace (`Prior-Labs/TabPFN-v2-clf`) to `~/.cache/tabpfn/`. Set `HTTPS_PROXY` if behind a corporate proxy. A local `model_dir` can be specified to use cached checkpoints offline (HPC).

---

## Experiment YAMLs

Primary-task only (no complication heads). The with-complications variant (`v1_with_complications.yaml`) is not listed here — it uses TabM and adds 7 complication output heads.

| File | Backbone | Notes |
|------|----------|-------|
| `v1_no_complications.yaml` | `mlp` | Baseline |
| `v1_no_complications_tabm.yaml` | `tabm` | TabM ensemble backbone |
| `v1_no_complications_saint.yaml` | `saint` | Column attention only (`style='col'`) |
| `v1_no_complications_tabpfn.yaml` | `tabpfn` | Pretrained in-context transformer |
| `v1_logistic_regression.yaml` | `lr` | Classical: scalar probability → `Linear(1→32)` |
| `v1_random_forest.yaml` | `rf` | Classical: scalar probability → `Linear(1→32)` (`use_tree_output: false`) |
| `v1_gradient_boosting.yaml` | `gbm` | Classical: scalar probability → `Linear(1→32)` (`use_tree_output: false`) |
| `v1_xgboost.yaml` | `xgboost` | Classical: scalar probability → `Linear(1→32)` |
| `v1_no_complications_ablation_no_cta.yaml` | `mlp` | CTA zeroed out — tabular-only ablation, Stage B removed |

Dedicated YAMLs exist mainly for SLURM job submission. Any backbone can also be selected at runtime without a dedicated file:
```bash
python scripts/train.py --config experiments/v1_no_complications.yaml \
    --set model.clinical_backbone=tabm
```
