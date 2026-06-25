# Configuration Reference

Experiments are defined by YAML files in `experiments/`. All share the same 2-stage training pipeline and `train.py`/`test.py` entry points.

| File | Backbone | Task |
|---|---|---|
| `mlp.yaml` | MLP | Reintervention prediction |
| `saint.yaml` | SAINT | Reintervention prediction |
| `tabm.yaml` | TabM | Reintervention prediction |
| `tabpfn.yaml` | TabPFN | Reintervention prediction |
| `v1_logistic_regression.yaml` | Logistic Regression | Reintervention prediction |
| `v1_random_forest.yaml` | Random Forest | Reintervention prediction |
| `v1_gradient_boosting.yaml` | Gradient Boosting | Reintervention prediction |
| `v1_xgboost.yaml` | XGBoost | Reintervention prediction |

---

## Full Annotated Config

```yaml
# ── Identity ──────────────────────────────────────────────────────────────────
name: v1_no_complications     # run identifier; used in W&B run name and checkpoint dir
version: 1

# ── Model ─────────────────────────────────────────────────────────────────────
model:
  complications: []           # [] → primary-task only
                              # List complications to enable the multi-task head, e.g.:
                              # ["endoleak_1a","endoleak_1b","endoleak_2","endoleak_3",
                              #  "endoleak_5","migration","infection"]
  aaa_dim: 5                  # number of AAA geometry metrics (do not change)
  clinical_backbone: mlp      # mlp (default) | tabm | lr | rf | gbm | xgboost
  backbone_cfg:               # hyperparameters forwarded to the chosen backbone
    # ── TabM fields (ignored for other backbones) ──────────────────────────
    k: 16                     # ensemble members (paper default 32; 16 for small data)
    n_blocks: 2               # MLP depth per ensemble member
    d_block: 64               # hidden width per member (halved from 128; ~4k params/positive)
    dropout: 0.3              # dropout inside each block
    num_embeddings: periodic  # periodic (default) | none
                              # periodic = PeriodicEmbeddings(lite=True): shared sinusoidal
                              # frequencies across all features; 48 params vs ~35 K for
                              # per-feature (lite=False) — right choice for 377 training samples
    d_embedding: 8            # embedding dim per feature (halved from 16 to match d_block)
    # ── Classical ML fields (for lr | rf | gbm | xgboost) ─────────────────
    # Use the sklearn/XGBoost constructor kwargs for the chosen backbone.
    # Examples: C, max_iter (LR); n_estimators, max_depth (RF/GBM/XGBoost).
    # scale_pos_weight is set automatically for XGBoost from training class counts.

# ── Training ──────────────────────────────────────────────────────────────────
training:
  seed: 3299                  # global seed for Python, NumPy, PyTorch, CUDA
  batch_size: 8               # samples per GPU batch
  # num_workers: DataLoader workers (default 4)
  image_size: [224, 224, 160] # spatial dimensions after preprocessing
  voxel_spacing: [1.5, 1.5, 3]    # resampling target (mm); matches Merlin's pretrained geometry
  hu_range: [-1000, 1000]      # HU clip range for CT intensity scaling

  # Loss parameters
  focal_gamma: 2.0            # focal loss exponent (down-weights easy negatives)
  label_smoothing: 0.02       # soft target smoothing (reduces overconfidence)
  pos_weight: 1.0             # BCE positive class weight; WeightedRandomSampler already
                              # balances batches 50/50, so pos_weight=1.0 avoids double-correction.
                              # Increase (e.g. to N_neg/N_pos) only if sampler is disabled.
  max_missing_pct: 0.5        # drop clinical features missing in ≥ this fraction of train patients
                              # reduces 240 raw features to ~42 genuinely informative ones;
                              # computed on train split only (no leakage); stored in checkpoint
  gate_l1_weight: 0.0001      # L1 regularisation weight on feature gate
                              # kept low since gate now operates on ~42 filtered features, not 240

  # Checkpoint selection & threshold
  recall_beta: 2.0            # β for F-beta threshold search over PR curve
                              # β=2 → recall weighted 4× over precision (F2-score)
  selection_metric: fbeta     # fbeta (default) | auroc
                              # fbeta: saves checkpoint on best val F2 (recall-focused)
                              # auroc: saves on best val AUROC (stable with small N positives)
                              # temperature calibration + F2 threshold search always use recall_beta

  # Stage B clinical freeze
  freeze_clinical_in_stage_b: false  # false (default): clinical branch continues training at 5e-5
                                     # true: clinical branch + feature_gate frozen after Stage A;
                                     #       Stage B becomes a visual-only adaptation pass

  # Modality dropout (applied in both Stage A and Stage B training)
  modality_dropout:
    p_cta: 0.0                # probability of zeroing CTA; set to 0 (CTA confirmed useful by ablation)
    p_aaa_given_cta: 0.15     # probability of zeroing AAA metrics when CTA is present
                              # (both p_aaa_given_cta and p_aaa_given_clin are checked
                              #  when CTA is present; AAA is kept when CTA is dropped)
    p_aaa_given_clin: 0.15    # additional probability of zeroing AAA when CTA is present
    p_clinical: 0.15          # probability of zeroing clinical features (non-causal regulariser)

  train_cache_rate: 0.5       # fraction of training data cached in RAM by MONAI CacheDataset
  val_cache_rate: 1.0         # val set fully cached (smaller, always fits in RAM)

  # XAI logging to W&B during training
  xai_every_n_epochs: 3       # run full_explain() on val samples every N epochs
  xai_samples: 4              # number of val patients to explain per logging run
  xai_gradcam: true           # also compute GradCAM (slower; requires I3ResNet backbone)

  # Training stages (run sequentially; Stage B loads Stage A's best checkpoint)
  stages:
    - name: A
      epochs: 60              # maximum epochs (early stopping may end it sooner)
      patience: 12            # stop if val F2-score does not improve for this many epochs
      T_0: 20                 # CosineAnnealingWarmRestarts restart period (epochs)

    - name: B
      epochs: 150
      patience: 20
      T_0: 30

# ── Paths ─────────────────────────────────────────────────────────────────────
paths:
  base_dir: /deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1
  checkpoint_dir: /deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/models/no_complications_MLP

# ── W&B ───────────────────────────────────────────────────────────────────────
wandb:
  project: AAA-Reintervention-Risk-MultiTab
  run_name: no_complications_MLP
  mode: online                # "online" | "offline" | "disabled"
```

---

## Overriding Values at Runtime

Any config value can be overridden with `--set` using dot-path notation. The CLI coerces integers, floats, and booleans automatically.

```bash
# Single override
python scripts/train.py --config experiments/mlp.yaml \
    --set training.batch_size=4

# Multiple overrides
python scripts/train.py --config experiments/mlp.yaml \
    --set training.batch_size=4 \
          training.stages[0].epochs=2 \
          wandb.mode=disabled \
          training.pos_weight=4.58
```

Coercion rules:
- `"4"` → `int(4)`
- `"0.001"` → `float(0.001)`
- `"true"` / `"false"` (any case) → `bool`
- Anything else → `str`

Nested keys use dot notation: `training.modality_dropout.p_cta=0.5`.

---

## Adding a New Experiment

1. Copy `experiments/mlp.yaml` to `experiments/my_experiment.yaml`
2. Update `name`, `version`, and `paths.checkpoint_dir`
3. Adjust any hyperparameters
4. Run:

```bash
python scripts/train.py --config experiments/my_experiment.yaml
```

There is no need to modify any source code — all hyperparameters are read from the config dict inside `run_pipeline`.
