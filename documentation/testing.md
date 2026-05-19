# Test Suite

## Running the Tests

```bash
cd /home/ranema/AI4EVAR/ai4evar
~/.conda/envs/ai4evar/bin/python -m pytest tests/ -v
```

220 tests, CPU-only, no GPU required, no external data files required, ~20 seconds.

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

- **CPU-only:** All tests run without a GPU. Spatial dimensions are tiny (6³ instead of 128³); batch size is 2.
- **No external data:** No NIfTI files, no SPSS files, no deepstore access. All data is synthesised in-memory using `tmp_path` fixtures or `torch.rand`.
- **Heavy-dependency mocks:** `merlin` and `peft` are stubbed out in `conftest.py` via `sys.modules` before any source import. The mock Merlin returns `(B, 2048)` zeros, matching the real backbone's output dimensionality, so the full model graph compiles and runs correctly.

---

## Test Files and Coverage

### `test_imports.py` — 13 tests

Smoke tests for every public symbol exported from each package's `__all__`. Catches typos in `__init__.py` exports, renamed functions, and circular import regressions immediately.

**Packages covered:** `src.models`, `src.data`, `src.training`, `src.xai`, `src.evaluation`

---

### `test_models.py` — 26 tests

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
- LoRA group LR is 5e-6 in Stage B
- `feature_gate` has the highest LR in Stage A
- Stage `"C"` raises `ValueError`

**`count_trainable`:**
- Correct total and trainable counts for `nn.Linear`, frozen parameters, nested `nn.Sequential`
- Works on bare `nn.Parameter` objects

**`print_model_param_overview`:**
- Runs without error
- Output contains `"clinical_branch"` and `"complications_head"`

---

### `test_loaders.py` — 40 tests

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

## Known Gaps

The following code paths exist in the production code but have no test coverage. This is not necessarily a problem — some are impractical to test without real data or a GPU — but they are worth noting.

### High Priority

| Gap | Why it matters |
|---|---|
| **`validate()`** (trainer.py) | The validation loop, metric collection, threshold computation, and W&B logging are exercised every training epoch but never in tests. A bug here would silently produce wrong AUC/threshold values. |
| **`run_test()`** (runner.py, 233 lines) | Produces all per-patient output files. Not tested beyond importability. A test with a dummy checkpoint and 2-patient synthetic split would cover the whole output pipeline. |
| **Checkpoint saving format** | Tests verify loading but nothing verifies that the training loop saves all expected keys (`threshold`, `feature_names`, `clin_mean`, `clin_std`, `complications`, `config`). A mismatch would only surface at inference time. |
| **F-beta threshold selection** | The `best_fbeta` / `best_threshold` PR-curve logic inside the validator is never called in tests. It is the threshold used at deployment. |

### Medium Priority

| Gap | Why it matters |
|---|---|
| **`build_loaders`** | Called every training run; tested for importability only. A test with a tiny synthetic data list would verify DataLoader structure and the balanced sampler. |
| **Gate L1 regularisation effect** | The feature gate shape is tested, but no test verifies that the L1 penalty actually changes the loss magnitude when `gate_l1_weight > 0`. |
| **FiLM clamp (±0.1)** | The architecture description says γ/β are clamped to ±0.1 via tanh, but this is not asserted in any test. A single forward-pass check would confirm the design is correctly implemented. |
| **W&B logging functions** | `log_val_table` and `log_xai_epoch` are completely untested. Mocking `wandb` with `MagicMock` and calling these functions would at least verify they don't raise. |

### Low Priority / Accepted Gaps

| Gap | Reason accepted |
|---|---|
| `run_pipeline` (full training loop) | Integration test territory; needs GPU and data; covered by SLURM runs |
| GradCAM with I3ResNet | Requires a different backbone class; mock always returns None |
| `_log_detailed_metrics` | Internal helper called inside `log_val_table`; low standalone risk |
