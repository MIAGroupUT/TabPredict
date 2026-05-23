# Test Suite

## Running the Tests

```bash
cd /home/ranema/AI4EVAR/ai4evar
~/.conda/envs/ai4evar/bin/python -m pytest tests/ -v
```

251 tests, CPU-only, no GPU required, no external data files required, ~17 seconds.

### On SLURM

The suite is designed to run on any SLURM allocation — including GPU nodes. The two trainer/runner test files patch `trainer.DEVICE` and `runner.DEVICE` to `cpu` via an autouse fixture, so they behave identically regardless of whether a GPU is available.

```bash
sbatch \
    --job-name=pytest \
    --output=/home/ranema/AI4EVAR/logs/pytest_%j.log \
    scripts/submit_slurm.sh train --config experiments/v1_no_complications.yaml
```

Or directly on a login/CPU node (no `sbatch` needed since there is no GPU dependency):

```bash
cd /home/ranema/AI4EVAR/ai4evar
~/.conda/envs/ai4evar/bin/python -m pytest tests/ -v
```

### Single file

```bash
~/.conda/envs/ai4evar/bin/python -m pytest tests/test_models.py -v
```

### Single test

```bash
~/.conda/envs/ai4evar/bin/python -m pytest tests/test_losses.py::test_focal_bce_perfect_prediction_lower -v
```

---

## Design Principles

- **CPU-only:** All tests run without a GPU. Spatial dimensions are tiny (6³ instead of 128³); batch size is 2. Where source modules select `cuda` at import time (trainer.py, runner.py), the tests patch `DEVICE` to `cpu` via an autouse fixture.
- **No external data:** No NIfTI files, no SPSS files, no deepstore access. All inputs — including checkpoints — are synthesised in-memory using `tmp_path` fixtures or `torch.rand`. Checkpoints contain randomly-initialised model weights, zero mean vectors, and synthetic feature names; no real patient data.
- **Heavy-dependency mocks:** `merlin` and `peft` are stubbed out in `conftest.py` via `sys.modules` before any source import. The mock Merlin returns `(B, 2048)` zeros, matching the real backbone's output dimensionality, so the full model graph compiles and runs correctly.

---

## Test Files and Coverage

### `test_imports.py` — 13 tests

Smoke tests for every public symbol exported from each package's `__all__`. Catches typos in `__init__.py` exports, renamed functions, and circular import regressions immediately.

**Packages covered:** `src.models`, `src.data`, `src.training`, `src.xai`, `src.evaluation`

---

### `test_models.py` — 29 tests

**Model constants:**
- `COMPLICATIONS_FULL` has 7 entries
- `COMPLICATIONS_NO_COMP` is empty
- `AAA_METRIC_NAMES` has 5 entries and contains expected keys

**Forward pass — no-complications variant:**
- Returns exactly 2 tensors `(logit, logvar)` of shape `(B, 1)`
- logvar is within [-4, 2] bounds
- All outputs are finite

**Forward pass — with-complications variant:**
- Returns exactly 4 tensors; `comp_logits` and `comp_logvar` have shape `(B, n_comp)`
- Same logvar clamp and finiteness checks

**`return_fused=True`:**
- Returns a `(B, 320)` tensor for both variants

**Model attributes:**
- No-comp model has no `complications_head`
- With-comp model stores the complication names
- `feature_gate` has correct shape `(clinical_dim,)`

**Helper functions:**
- `get_model` unwraps `DataParallel`; passes through plain models unchanged
- `disable_checkpointing` sets `gradient_checkpointing=False` on modules that have it

**`load_checkpoint`:**
- Normal load + returns checkpoint dict with expected keys
- Strips `"module."` prefix from DataParallel-saved checkpoints
- Filters unexpected legacy keys with `strict=False`
- Raises `RuntimeError` on missing keys with `strict=True`

**FiLM clamp (±0.1):**
- Multiplicative scale `1 + tanh(γ) × 0.1` saturates to 1.1 with extreme positive γ
- Additive shift `tanh(β) × 0.1` saturates to +0.1 with extreme positive β (zero input)
- Additive shift saturates to −0.1 with extreme negative β (zero input)

---

### `test_losses.py` — 18 tests

**`_focal_bce_per_sample`:**
- Output shape preserved
- Non-negative outputs
- Correct prediction has lower loss than wrong prediction
- Label smoothing increases loss for confident predictions

**`_heteroscedastic`:**
- Returns a scalar
- Finite output
- Positive output
- Handles extreme logvar without NaN

**`ReinterventionLoss`:**
- Returns `(loss_scalar, details_dict)`
- `details_dict` contains `loss_total`, `loss_reint`, `mean_uncertainty`
- `loss_total == loss_reint` (no complications term)
- Handles both `(B, 1)` and `(B,)` GT shapes

**`MultiTaskReinterventionLoss`:**
- Returns `(loss_scalar, details_dict)`
- `details_dict` contains `loss_total`, `loss_reint`, `loss_comp`, `mean_uncertainty`
- All-missing GT (`-1`) produces `loss_comp == 0.0` exactly
- Partial masking produces `loss_comp > 0`
- Higher `comp_weight` increases total loss
- All outputs finite on random inputs

---

### `test_stages.py` — 37 tests

**`set_all_seeds`:**
- Same seed → same random sequence
- Different seeds → different sequences

**`causal_modality_dropout`:**
- Shape preserved for all three inputs
- `p=0` → outputs equal inputs exactly
- `p_cta=1` → all CTA samples zeroed
- `p_clinical=1` → all clinical samples zeroed without touching CTA or AAA

**`maybe_apply_modality_dropout`:**
- `is_training=False` → inputs returned unchanged (same object)
- `is_training=True` with `p_cta=1` → CTA is zeroed

**Stage A freeze/train pattern:** Verifies individually for `visual_branch`, `cta_adapter`, `visual_proj` (frozen), and `clinical_branch`, `aaa_metrics_branch`, `reintervention_head`, `feature_gate`, `complications_head` (trainable).

**Stage B:** `condition_net`, `clinical_branch`, `cta_adapter`, `visual_proj` are trainable after `set_stage_b`.

**`build_optimizer`:**
- Returns `AdamW`
- Correct group count for Stage A (6 no-comp, 7 with-comp)
- Correct group count for Stage B (9 no-comp, 10 with-comp)
- `i3resnet_bn` group LR is 1e-5 in Stage B
- `feature_gate` has the highest LR in Stage A
- Stage `"C"` raises `ValueError`

**`count_trainable`:**
- Correct total and trainable counts for `nn.Linear`, frozen parameters, nested `nn.Sequential`
- Works on bare `nn.Parameter` objects

**`print_model_param_overview`:**
- Runs without error
- Output contains `"clinical_branch"` and `"complications_head"`

---

### `test_loaders.py` — 44 tests

**Clinical feature cleaning:**
- Numeric strings → float
- `None` → 0.0
- Empty string → 0.0
- Invalid string → 0.0
- Float passthrough
- All dictionary keys preserved

**`load_aaa_metrics`:**
- Missing file → returns all-zero dict
- Valid JSON → correct values
- Partial JSON (missing keys) → missing keys are 0.0
- Returns exactly 5 float values

**`load_splits_and_config`:**
- Returns train/val/test split lists
- Feature names are sorted alphabetically
- Correct feature count from the config JSON
- `None`-valued features are cleaned to 0.0

**`compute_clin_stats`:**
- Mean is correct
- Std is positive
- Constant feature → std of 1.0 (safe fallback, not zero division)
- Missing key in a patient record → defaults to 0.0

**`build_data_list`:**
- Returns a list of dicts
- Clinical vector has correct shape
- AAA vector defaults to zeros when `metricsTr` file is missing
- Reintervention label is correctly read
- Complications are included only when `complications != []`
- No `"comp"` key when `complications == []`

**MONAI transforms:**
- `BuildCTAd` produces correct output key and shape `(4, D, H, W)`
- Each channel maps to the correct source (CT, lumen, ILT, distance)
- `NormalizeClinicald` produces correct z-score values and tensor output
- Zero-std features are handled safely (no NaN/inf)

**`RuntimeDataset`:**
- `len()` is correct
- `__getitem__` applies the provided transform
- Identity transform returns unchanged data

**Transform factories:**
- `get_val_runtime_transforms` and `get_train_runtime_transforms` return callables

**Tensor keys:**
- Without complications: `"ctad"`, `"clin"`, `"aaa"`, `"reint"` are present; no `"comp"` key
- With complications: `"comp"` key is also present

**`build_loaders`:**
- Returns a `(train_loader, val_loader)` tuple
- Train DataLoader is built with `shuffle=True` (natural class distribution; imbalance handled via `pos_weight` in the loss)
- Val DataLoader is built with `shuffle=False` (deterministic evaluation order)
- Requested `batch_size` is forwarded to both loaders

`CacheDataset` and `torch.utils.data.DataLoader` are mocked so no NIfTI files or GPU workers are needed.

---

### `test_metrics.py` — 30 tests

**`label_type`:** All four outcomes (TP, FP, TN, FN) correctly classified.

**`compute_overall_metrics` — structure:** Required keys present, `n_samples` correct, `n_positive`/`n_negative` correct.

**`compute_overall_metrics` — numeric correctness:**
- Perfect predictions → AUC=1.0, accuracy=1.0
- All-wrong predictions → accuracy=0.0
- Confusion matrix entries (TP, FP, TN, FN) correct
- `threshold_used` matches input
- AUC in [0, 1] on random inputs
- All-same GT → AUC=0.5 (graceful undefined-AUC handling)

**`compute_overall_metrics` — with complications:**
- `comp_auc_detail` and `comp_mean_auc` present when complication args provided
- AUC values correct (1.0 for perfect complication predictions)
- All-missing GT (`-1`) → no `comp_auc_detail` key
- No complication args → no complication keys in result

**`save_metrics`:**
- Creates `test_metrics.json` and `test_metrics.csv`
- JSON contains correct values
- CSV has header `["metric", "value"]`
- `comp_auc_detail` is expanded to individual `auc_<comp_name>` rows in CSV

**`format_xai_report`:**
- Returns a non-empty string
- Contains patient ID, probability, top clinical feature name, label type
- Complication section present only when complications provided

---

### `test_xai.py` — 34 tests

**Setup:** `ReinterventionXAI` stores feature names and complications correctly.

**`_confidence_level`:** HIGH (≥0.75), MED (≥0.50), LOW (<0.50), boundary values.

**`_get_i3res`:** Returns `None` for the mock backbone (no I3ResNet).

**`clinical_importance`:**
- Returns a list
- Each element is a 3-tuple `(name, score, level)`
- At most 20 entries
- Feature names are valid (from the provided list)

**`aaa_importance`:**
- Returns exactly 5 entries
- Each is a 3-tuple
- Names are the known AAA metric names

**`modality_importance`:**
- Contains keys `"CTA"`, `"Clinical"`, `"AAA_metrics"`
- Values sum to 1.0
- All values non-negative

**`full_explain` — no-comp:**
- Contains all expected keys
- `reintervention_prob` in [0, 1]
- `reintervention_confidence` in [0, 1]
- `confidence_level` is a valid string
- No complication keys present

**`full_explain` — with-comp:**
- Contains `complication_probs` and `complication_confidence`
- Complication names match those provided to the constructor
- Probabilities in [0, 1]
- Confidence levels are valid strings

**GradCAM:**
- Returns `None` for both model variants (mock backbone is not I3ResNet)

**`save_gradcam_pngs`:**
- Creates expected files in `tmp_path`
- File names are correct

**`_render_slice`:** Runs without error.

---

### `test_scripts.py` — 22 tests

**`_apply_overrides` — type coercion:**
- `"4"` → `int(4)`, `"0.0001"` → `float`, `"true"/"false"` → `bool`
- Integer coercion wins over float for whole-number strings
- String fallback for uncoercible values
- Nested dot-path keys create intermediate dicts
- Top-level keys work
- Empty override list is a no-op
- Multiple overrides applied in one call
- Returns the same dict object (mutates in place)

**`train.py` argparse:**
- Missing `--config` raises `SystemExit` with non-zero code
- Valid `--config` parses correctly
- `--set` accumulates a list of override strings

**`test.py` argparse:**
- Missing any required argument (`--config`, `--checkpoint`, `--output_dir`) raises `SystemExit`
- All required args present → parses correctly
- Default values for `--split`, `--gradcam`, `--workers` are correct
- Invalid `--split` choice raises `SystemExit`
- `--gradcam` flag sets to `True`

---

### `test_trainer.py` — 19 tests

Tests for `validate()`, gate L1 regularisation, W&B logging helpers, and the checkpoint save format.

**SLURM/GPU safety:** An autouse fixture patches `src.training.trainer.DEVICE` to `cpu`, so the tests run identically on CPU-only and GPU-allocated nodes.

**`validate()` — no-complications variant:**
- Returns `(float, dict)`
- All 16 required metric keys are present (`val/loss_total`, `val/reint_auc`, `val/reint_best_thresh`, confusion matrix entries, rates, ...)
- No complication keys leak into the no-comp result
- AUC, accuracy, and threshold are in [0, 1]; loss is non-negative
- F-beta threshold (the deployment threshold) is in [0, 1]
- All-same ground-truth produces AUC = 0.5 (graceful undefined-AUC handling)
- `wandb.log` is called exactly once per invocation

**`validate()` — with-complications variant:**
- `val/comp_mean_auc` is present and in [0, 1]
- Standard reintervention keys are still emitted alongside complication keys

**Checkpoint save format:**
- Dict produced by trainer.py contains all 7 required keys: `state_dict`, `threshold`, `feature_names`, `clin_mean`, `clin_std`, `complications`, `config`
- `threshold`, `feature_names`, `clin_mean`, `clin_std` survive a save/load round-trip correctly
- With-complications checkpoint stores the complication name list

**Gate L1 regularisation effect:**
- Adding `gate_l1_weight × sigmoid(feature_gate).sum()` with weight > 0 produces a strictly larger total loss than the criterion alone
- With `gate_l1_weight = 0` the added term is exactly zero (no numerical drift)

**W&B logging helpers (`log_val_table`, `log_xai_epoch`):**
- Both run without error for the no-complications variant (wandb mocked)
- Both run without error for the with-complications variant
- `wandb.log` is called exactly once per invocation in each case

---

### `test_runner.py` — 6 tests

Tests for `run_test()` (evaluation/runner.py) — the full inference pipeline.

**SLURM/GPU safety:** An autouse fixture patches `src.evaluation.runner.DEVICE` to `cpu`.

**Data loading:** `load_splits_and_config`, `build_data_list`, `get_cached_transforms`, `get_val_runtime_transforms`, `CacheDataset`, and `DataLoader` are mocked. The model and checkpoint are real (randomly-initialised weights, no real patient data).

**No-complications variant (2 synthetic patients):**
- Top-level files created: `test_metrics.json`, `test_metrics.csv`, `test_log.txt`
- Per-patient directories contain `prediction.json`, `xai.json`, `xai_report.txt`
- `prediction.json` has all required keys (`pid`, `reintervention_prob`, `reintervention_pred`, `reintervention_gt`, `label_type`, `threshold`, `reint_logvar`) with valid values
- `test_metrics.json` contains `n_samples`, `auc`, `f1`, `precision`, `recall`, `accuracy`, `threshold_used`; `n_samples == 2`

**With-complications variant (1 synthetic patient):**
- Per-patient files created
- `prediction.json` contains `complication_probs` dict keyed by complication name; all values in [0, 1]

---

## Known Gaps

### Accepted Gaps

All high- and medium-priority gaps have been addressed. The remaining gaps are integration-level concerns that require real data or a GPU and are intentionally left uncovered.

| Gap | Reason accepted |
|---|---|
| `run_pipeline` (full training loop) | Integration test territory; needs GPU and data; covered by SLURM runs |
| GradCAM with I3ResNet | Requires a different backbone class; mock backbone always returns None |
| `_log_detailed_metrics` | Thin internal helper; called and implicitly exercised inside `validate()` |
