# Configuration Reference

Experiments are defined by YAML files in `experiments/`. There are two variants:

| File | Model |
|---|---|
| `v1_no_complications.yaml` | Reintervention prediction only (primary task) |
| `v1_with_complications.yaml` | Reintervention + 7 complication heads (multi-task) |

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

# ── Training ──────────────────────────────────────────────────────────────────
training:
  seed: 3299                  # global seed for Python, NumPy, PyTorch, CUDA
  batch_size: 8               # samples per GPU batch
  num_workers: 16             # DataLoader worker processes
  image_size: [128, 128, 128] # spatial dimensions after preprocessing
  voxel_spacing: [1.5, 1.5, 1.5]  # isotropic resampling target (mm)

  # Loss parameters
  focal_gamma: 2.0            # focal loss exponent (down-weights easy negatives)
  label_smoothing: 0.02       # soft target smoothing (reduces overconfidence)
  pos_weight: 4.58            # BCE positive class weight = N_neg/N_pos (302/66 in training set)
  gate_l1_weight: 0.0001      # L1 regularisation weight on feature gate

  # Threshold selection
  recall_beta: 2.0            # β for F-beta threshold search over PR curve
                              # β > 1 → recall weighted more than precision

  # Modality dropout (Stage B training only)
  modality_dropout:
    p_cta: 0.10               # probability of zeroing CTA for a batch sample
    p_aaa_given_cta: 0.15     # probability of zeroing AAA when CTA is already dropped
    p_aaa_given_clin: 0.15    # probability of zeroing AAA when CTA is present
    p_clinical: 0.20          # probability of zeroing clinical features (non-causal regulariser)

  # XAI logging to W&B during training
  xai_every_n_epochs: 3       # run full_explain() on val samples every N epochs
  xai_samples: 4              # number of val patients to explain per logging run
  xai_gradcam: true           # also compute GradCAM (slower; requires I3ResNet backbone)

  # Training stages (run sequentially; Stage B loads Stage A's best checkpoint)
  stages:
    - name: A
      epochs: 60              # maximum epochs (early stopping may end it sooner)
      patience: 12            # stop if val AUC does not improve for this many epochs
      T_0: 20                 # CosineAnnealingWarmRestarts restart period (epochs)

    - name: B
      epochs: 150
      patience: 20
      T_0: 30

# ── Paths ─────────────────────────────────────────────────────────────────────
paths:
  base_dir: /deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1
  cache_dir: /deepstore/datasets/mia/AI4EVAR/derivatives/monai_cache
  checkpoint_dir: /home/ranema/AI4EVAR/checkpoints/v1_no_complications

# ── W&B ───────────────────────────────────────────────────────────────────────
wandb:
  project: AAA-Reintervention-Risk
  run_name: v1_no_complications
  mode: online                # "online" | "offline" | "disabled"
```

---

## Multi-Task Variant Differences

The `v1_with_complications.yaml` adds complications and adjusts the comp loss weight per stage:

```yaml
model:
  complications:
    - endoleak_1a
    - endoleak_1b
    - endoleak_2
    - endoleak_3
    - endoleak_5
    - migration
    - infection

training:
  # comp_weight scales the complication loss relative to the reintervention loss
  # Typically lower in Stage A (tabular warm-up) and higher in Stage B
  comp_weight_stage_a: 0.1
  comp_weight_stage_b: 0.3
```

---

## Overriding Values at Runtime

Any config value can be overridden with `--set` using dot-path notation. The CLI coerces integers, floats, and booleans automatically.

```bash
# Single override
python scripts/train.py --config experiments/v1_no_complications.yaml \
    --set training.batch_size=4

# Multiple overrides
python scripts/train.py --config experiments/v1_no_complications.yaml \
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

1. Copy `experiments/v1_no_complications.yaml` to `experiments/my_experiment.yaml`
2. Update `name`, `version`, and `paths.checkpoint_dir`
3. Adjust any hyperparameters
4. Run:

```bash
python scripts/train.py --config experiments/my_experiment.yaml
```

There is no need to modify any source code — all hyperparameters are read from the config dict inside `run_pipeline`.
