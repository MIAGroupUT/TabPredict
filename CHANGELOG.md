# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Entries are grouped by push milestone (no formal version tags yet).

---

## [Unreleased]

_No unreleased changes._

---

## [2026-07-10] — Fix notebook matrix calculation of auc

### Fixed

- **`compare_results.ipynb`** — Matrix plot now shows exact same AUC numbers as ROC. Before it avergaed `stage_a` and `stage_b` AUC values which lead to slightly different AUCs. This is now fixed, it only uses `stage_b` AUC.

---

## [2026-07-03] — Restore UMCG site; revert selection_metric to fbeta

### Fixed

- **`00_dicom_to_nifti_renaming.py`** — `convert_to_nifti()` now reorients the read DICOM series to canonical RAS via `sitk.DICOMOrient(img, "RAS")` before saving. Raw NIfTIs previously kept whatever direction cosines the source scanner/site produced, inconsistent across patients; now consistent with the `Orientationd(axcodes="RAS")` reorientation the MONAI pipeline (`src/data/loaders.py`) applies at load time, so downstream steps (nnUNet, metrics, CFD) work on already-RAS-consistent volumes.

### Changed

- **UMCG site restored** — `03_data_matching_v1.py` re-enables the `"UMCG"` entry in `SITE_CONFIG` (was commented out); output filenames reverted from `_no_umcg` back to plain `radar_multimodal_config_v1_multitab.json` across `03_data_matching_v1.py`, `04_data_organizing.py`, `07_generate_data_split.py`, `src/data/loaders.py`, and `tests/test_loaders.py`.
- **`selection_metric`** reverted `auroc` → `fbeta` (default) across all 8 experiment YAMLs; `checkpoint_dir` moved `models_no_umcg_auroc/` → `models_no_umcg_fbeta/<NAME>` (TabM corrected from stray `models_umcg_fbeta/`); `run_name` gets `_no_umcg_fbeta` suffix.
- **`src/training/trainer.py`** — `validate()`'s discarded third return value now captured as `(val_rp, val_rg)`; best-checkpoint threshold read once into `save_thresh` and reused for the log line; added inline comment explaining the `auroc` vs `fbeta` selection tradeoff.
- **`scripts/submit_slurm.sh`** — `--constraint` widened from `l40` to `"l40|l40s|a40|rtx6000pro|h200nvl"` to reduce queue wait.
- **`requirements.txt`** — dropped stray `pip install git+https://github.com/s-marton/GRANDE.git --no-deps` line after `nnunetv2==2.7.0`.

---

## [2026-06-25] — Switch to no-UMCG data; rename YAMLs; ROC grid; PDF plots

### Added

- **`eval/compare_results.ipynb`** — 4×2 ROC grid cell: individual per-method subplot (no title, legend shows name + AUC), explicit row ordering (MLP, SAINT, TabM, TabPFN top; LR, RF, XGBoost, GB bottom). Saved as `ROC_grid.pdf`.
- All plot outputs switched from PNG (raster) to **PDF** (vector) for publication quality: `ROC_all`, `Metric_comparison_bar_chart`, `AUROC_bootstrap`, `AUROC_Matrix`, `ROC_grid`, and per-run split/stage comparison plots.

### Changed

- **Experiment YAMLs** — renamed `v1_no_complications_<name>.yaml` → `<name>.yaml` (MLP, TabM, SAINT, TabPFN); classical YAMLs dropped `v1_` prefix. `run_name` and `checkpoint_dir` cleaned up to short names. `selection_metric` changed `fbeta` → `auroc` across all 8 configs. `checkpoint_dir` paths updated to `models_no_umcg_auroc/<NAME>`.
- **`src/data/loaders.py`** — pipeline now reads `dataset_splits_multitab_no_umcg.json` and `radar_multimodal_config_v1_multitab_no_umcg.json`, excluding UMCG patients (zero re-intervention events) from training.
- **`tests/test_loaders.py`** — fixture filenames updated to `_no_umcg` variants; fixes 5 previously failing tests.

---

## [2026-06-24] — Add selection_metric flag; remove generate_reports; eval label cleanup

### Added

- **`selection_metric`** — new `training:` YAML key (`fbeta` | `auroc`, default `fbeta`). Controls which val metric drives checkpoint saving and early stopping. `fbeta` preserves previous F2-based behaviour; `auroc` uses threshold-free AUROC as a more stable stopping signal with small positive counts. Temperature calibration and F2 threshold search always use `recall_beta` regardless of this setting. Added to all 8 experiment YAMLs and `trainer.py`.

### Changed

- **`eval/compare_results.ipynb`** — model display labels shortened: "Logistic Regression" → "LR", "Random Forest" → "RF", "GradientBoosting" → "GB".

### Removed

- **`preprocessing/generate_reports/`** — deleted LLM-based radiology report generation scripts (`generate_reports.py`, `run_reports.sh`, `test_mistral_download.py`); Mistral-based report generation is no longer part of the pipeline.

---

## [2026-06-22] — MICCAI MultiTab workshop branch setup

### Added

- **`miccai_multitab` branch** — dedicated branch for the MICCAI MultiTab workshop submission, branched from `c046247`.

### Changed

- **`03_data_matching_v1.py`** — Vascutek Anaconda device type (P_IP_P_DT=9) no longer excluded; all device types are now included in the matched dataset.
- **Output filenames** — preprocessing and training pipeline now writes `_multitab`-suffixed files to avoid overwriting the v1 dataset: `radar_multimodal_config_v1_multitab.json`, `radar_multimodal_config_v1_multitab_analysis.txt`, `dataset_splits_multitab.json`. Affects `03_data_matching_v1.py`, `04_data_organizing.py`, `07_generate_data_split.py`, and `src/data/loaders.py`.
- **Experiment `run_name` fields** — date suffix `_17062026` removed from all 9 experiment YAMLs for clean W&B run naming. `v1_` prefix also stripped from `run_name` and `checkpoint_dir` values in all 10 configs (e.g. `v1_no_complications_MLP` → `no_complications_MLP`).
- **All output paths consolidated under deepstore** — logs → `/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/`, preprocessing logs → `.../logs/preprocessing_v1_multitab/`, models → `.../models/`, test results → `.../test_results/`. Updated in `submit_slurm.sh`, `prepare_data.sh`, and all experiment YAMLs.
- **`submit_slurm.sh`** — `cd` path updated to `ai4evar_multitab`; example checkpoint and test-result paths reflect new deepstore layout.
- **`documentation/data.md`** — all filename references updated to `_multitab` variants.
- **`tests/test_loaders.py`** — fixture filenames updated to `_multitab` suffix (5 tests were failing after loader path change).

### Removed

- **Weight EMA** (`weight_ema: 0.99`) — not present on this branch; state reflects `c046247` which predates the EMA + MC-dropout + AUROC-selection additions from `4083c1f`.

---

## [2026-06-17] — Freeze clinical in Stage B; fix tests; bump experiment dates

### Added

- **`freeze_clinical_in_stage_b`** — new boolean flag in experiment YAML `training:` block. When `true`, the clinical branch and `feature_gate` are fully frozen during Stage B so only the visual pathway (cta_adapter, visual_proj, I3ResNet BN) and fusion/heads continue updating. Default `false` preserves the previous joint-training behaviour. All 10 experiment YAMLs set to `true` for the 17062026 runs.
- **`set_stage_b(freeze_clinical=False)`** — `stages.py` now accepts `freeze_clinical` kwarg; when `True` skips enabling `requires_grad` on `clinical_branch` and `feature_gate`.
- **`build_optimizer(freeze_clinical=False)`** — Stage B param groups omit `clinical` and `feature_gate` when `freeze_clinical=True` (6 groups instead of 8).
- **4 new `test_stages` tests** — cover `freeze_clinical=True`: branch/gate frozen, visual still trains, optimizer group count (6) and group-name exclusion.
- **`documentation/training.md`** — Stage B table split into two variants (`freeze_clinical=false` / `true`); optimizer group count tables corrected (removed stale `aaa` and `i3resnet_layer4` groups; Stage A 5 groups, Stage B 8/6).
- **`documentation/configuration.md`** — `freeze_clinical_in_stage_b` key added with inline comments.

### Changed

- **All 10 experiment YAMLs** — `run_name` / `checkpoint_dir` date suffix bumped `15062026` → `17062026`. `v1_with_complications` also gets date suffix in `run_name` (was undated) and path moved to deepstore.
- **All 10 experiment YAMLs** — `freeze_clinical_in_stage_b: true` set for 17062026 training runs.
- **`v1_xgboost.yaml`** — `use_tree_output: false` added explicitly to `backbone_cfg` to enforce scalar-prob → `Linear(1→32)` projection (avoids high-dim TreeSHAP path on small N).

### Fixed

- **`tests/test_loaders.py`** — 12 `build_data_list` / `build_loaders` tests failing because test configs lacked `scans.pre_op.nifti`; added `"scans": {"pre_op": {"nifti": "dummy.nii.gz"}}` stub to all affected configs. All 420 tests now pass.

### Removed

- **`experiments/v1_no_complications.yaml`** — original MLP baseline config deleted; superseded by **`experiments/v1_no_complications_mlp.yaml`** (same hyperparameters, updated checkpoint path/run name).

---

## [2026-06-15] — Classical backbone scalar projection (15062026)

### Added

- **`ClassicalBranchWrapper.use_tree_output`** (`bool`, default `False`) — new flag in `backbone_cfg` controlling the projection input for tree-based classical backbones. When `False` (default), all backends use a single calibrated probability → `Linear(1→32)` (64 params). When `True`, each backend exposes a richer structural representation: `rf`/`gbm` → per-tree output → `Linear(n_est→32)` (~9.6k params), `xgboost` → TreeSHAP per feature → `Linear(n_feat→32)` (~1.5k params). The default changed to `False` because on RADAR (~700 training samples) the wide projection has a 14:1 parameter-to-sample ratio and overfits, producing ~10% lower test AUC than the scalar approach.
- **`03_data_matching_v1.py` — Vascutek Anaconda exclusion**: patients where `P_IP_P_DT == 9` are now excluded before matching. New `anaconda_excluded` counter in `stats`; analysis report prints count and per-site patient list.
- **`03_data_matching_v1.py` — per-FU SAC change extraction**: `extract_ground_truth()` now computes `sac_per_fu` — per-FU-visit list of `{fu_num, date, diameter_mm, modality, diff_from_bl_mm, category}`. Also stores `baseline_diameter_mm` and `baseline_modality`.
- **`03_data_matching_v1.py` — SAC CHANGE PER FOLLOW-UP VISIT report section** (console + file): one row per FU number with N measured and regression/stable/expansion counts.
- **`03_data_matching_v1.py` — SAC PER-PATIENT DETAIL section** (file only): per-patient stacked table with op date + baseline then one row per FU visit. REGRESSION/EXPANSION uppercased.

### Changed

- **`experiments/v1_gradient_boosting.yaml`** — `checkpoint_dir` / `run_name` suffix `10062026` → `15062026`; `use_tree_output: false` added to `backbone_cfg`.
- **`experiments/v1_random_forest.yaml`** — `use_tree_output: false` added to `backbone_cfg` (documented flag, existing behaviour already matched scalar default).
- **`documentation/models.md`** — backbone comparison table footnote added for `use_tree_output`; projection section rewritten with two sub-tables (scalar default vs. tree output); YAML table Notes column updated to reflect scalar default for rf/gbm/xgboost; experiment YAML table expanded with all 9 primary-task configs.
- **`03_data_matching_v1.py`** — existing `sac_regression` / `sac_stable` / `sac_expansion` ground-truth flags unchanged (baseline vs last FU); report clarifies with a note pointing to the new per-FU section.
- **`pytest.ini`** — merged `testpaths = tests` and `addopts` from now-deleted `.pytest.ini`. Single config file; behaviour unchanged.

### Removed

- **`.pytest.ini`** — duplicate config deleted; consolidated into `pytest.ini`.

### Tests

- **`test_wrapper_projection_width`** split into `test_wrapper_projection_width_tree` (asserts wide projection when `use_tree_output=True`) and `test_wrapper_projection_width_scalar` (asserts `in_features=1` for all backends at default).
- **`test_extract_features_shape`** split into `test_extract_features_shape_tree` (`use_tree_output=True`, wide output) and `test_extract_features_shape_scalar` (`use_tree_output=False`, shape `(B, 1)` for all backends).

---

## [2026-06-10] — Richer classical projections; RF/GBM/XGBoost reruns; docs index → Checkpointing 10062026

### Added

- **`documentation/README.md`** — new navigation index for the `documentation/` folder: quick-start routing guide ("new to the project? → Architecture → Training → Data"), grouped index table for all 11 docs, and a key-numbers quick-reference block.
- **`ClassicalBranchWrapper._preprocess()`** — new helper applying `_impute()` + `StandardScaler.transform()`, mirroring the sklearn pipeline prefix. Shared by `_extract_features()` and GBM leaf scale computation at fit time.
- **`ClassicalBranchWrapper._extract_features()`** — new method dispatching per `model_type` to return the richer representation fed into the projection layer.

### Changed

- **`ClassicalBranchWrapper`** projection input is no longer a single scalar for tree models. Each backbone now exposes a structurally richer representation:
  - **`rf`**: per-tree positive-class probability `(B, n_estimators)` → `Linear(300→32)` (was `Linear(1→32)`)
  - **`gbm`**: per-tree normalised leaf node index `(B, n_estimators)` → `Linear(300→32)` (was `Linear(1→32)`)
  - **`xgboost`**: TreeSHAP values per input feature `(B, n_features)` → `Linear(n_feat→32)` (was `Linear(1→32)`)
  - **`lr`**: unchanged — scalar `predict_proba` `(B, 1)` → `Linear(1→32)`
- **`ClassicalBranchWrapper.fit()`** — GBM path now computes `_gbm_leaf_scale` (per-tree max leaf index, clipped to ≥ 1.0) from training data and stores it on `baseline` so `joblib.dump` serialises it into the checkpoint automatically.
- **`experiments/v1_random_forest.yaml`** — `run_name` / `checkpoint_dir` updated `04062026` → `10062026`; job 512483 submitted (first run with `Linear(300→32)` RF projection).
- **`experiments/v1_gradient_boosting.yaml`** — `08062026` → `10062026`; job 512484 replaces stale run 512463 (which used old `Linear(1→32)`).
- **`experiments/v1_xgboost.yaml`** — `08062026` → `10062026`; job 512485 submitted (first run with SHAP `Linear(n_feat→32)` projection).
- **`documentation/models.md`** — projection section rewritten with a per-model representation table (shape, method, projection); comparison table updated with corrected projection widths and param counts; GBM leaf scale serialisation note added.
- **`README.md`** — documentation table: `models.md` and `preprocessing.md` added as explicit rows.

### Tests

- `test_wrapper_projection_receives_gradients` docstring updated (removed stale `Linear(1→32)` reference).
- **`test_wrapper_projection_width`** — new parametrised test asserting `projection.in_features` matches per-model expected width (`lr=1`, `rf=300`, `gbm=300`, `xgboost=N_FEAT`).
- **`test_gbm_leaf_scale_stored_after_fit`** — asserts `_gbm_leaf_scale` exists on `baseline`, shape `(300,)`, all ≥ 1.0.
- **`test_extract_features_shape`** — asserts `_extract_features()` returns `(B, expected_cols)` float32 for lr / rf / gbm.

---

## [2026-06-10] — Differentiable TabPFN; models.md; GBM run; fix checkpoint deserialization → Checkpointing 10062026

### Added

- **`documentation/models.md`** — new technical reference covering all eight `clinical_branch` backbones (mlp, tabm, saint, lr, rf, gbm, xgboost, tabpfn). Includes a comparison table (differentiability, feature gate, imputation, projection type, fit timing, param count), per-model imputation breakdown (XGBoost no-op vs. median for others), projection rationale, execution flow per backbone type, and hyperparameter tables.
- **`documentation/preprocessing.md`** — new 294-line technical walkthrough of the full v1 preprocessing pipeline (`preprocessing/case_matching_v1/`, steps 00–07): DICOM→NIfTI conversion, SPSS matching, nnUNet segmentation + AAA metrics generation, full-body crop logic, stratified split generation.
- **`TabPFNBranchWrapper.n_trainable_layers`** (`int`, default `2`) — last N encoder/decoder transformer blocks kept trainable during Stage A + B. Set `0` for projection-only training. Implemented via `_setup_diff_mode()`: after `FinetunedTabPFNClassifier.fit()`, freezes all TabPFN params then selectively unfreezes the last N blocks of `transformer_decoder` (or `transformer_encoder` when no decoder).
- **`TabPFNBranchWrapper._setup_diff_mode()`** — sets `differentiable_input=True` on `finetuned_inference_classifier_`, calls `fit_with_differentiable_input(X_train, y_train)` to initialise `InferenceEngineCachePreprocessing(inference_mode=False)`, then configures `requires_grad` per layer. Called by both `fit()` and `restore_from_bytes()`.
- **`TabPFNBranchWrapper.restore_from_bytes(data)`** — instance method replacing the broken static `deserialize()`. Restores `_clf` via joblib and immediately re-runs `_setup_diff_mode()` so the differentiable executor is live after checkpoint load.
- **`TabPFNBranchWrapper._manages_own_grad = True`** — class flag read by `set_stage_a` / `set_stage_b` in `stages.py` to prevent clobbering the per-layer `requires_grad` flags configured by `_setup_diff_mode()`.

### Changed

- **`TabPFNBranchWrapper._is_non_differentiable`** `True` → `False`. TabPFN is now partially differentiable (last N blocks + projection). Consequence: `is_classical_backbone()` now returns `False` for TabPFN; the L1 **feature gate is trained** alongside the unfrozen blocks.
- **`TabPFNBranchWrapper.forward()`** — replaced numpy path (`predict_raw_logits(x_np)` via `.detach().cpu().numpy()`) with differentiable tensor path: `diff_clf.forward(x, use_inference_mode=True, return_raw_logits=True)`. Gradient preserved because `TabPFNClassifier` sets `actual_inference_mode = use_inference_mode and not differentiable_input = False` when `differentiable_input=True`.
- **`set_stage_a` / `set_stage_b`** (`stages.py`) — clinical branch params are no longer unconditionally enabled. When `branch._manages_own_grad` is `True`, only `branch.projection.requires_grad_(True)` is called; backbone layer grad flags are left as configured by `_setup_diff_mode()`.
- **`load_checkpoint`** (`reintervention.py`) — fixed broken TabPFN deserialization. Previously only `ClassicalBranchWrapper` was handled; `TabPFNBranchWrapper` checkpoints were silently ignored (branch stayed unfitted after load). Now uses duck-typing `hasattr(branch, "restore_from_bytes")` to restore TabPFN branches.
- **`experiments/v1_no_complications_tabpfn.yaml`** — `checkpoint_dir` and `run_name` updated from `08062026` → `10062026` for the new differentiable run.
- **`experiments/v1_gradient_boosting.yaml`** — `checkpoint_dir` / `run_name` updated `04062026` → `08062026`. GBM backbone now submitted for the first time (job 512463, `10062026` run via `--set` overrides).
- **`documentation/architecture.md`** — clarified 4-channel CTA description; added cross-link to `preprocessing.md` for AAA metrics derivation.
- **`documentation/data.md`** — preprocessing pipeline section replaced with quick-reference table + link to `preprocessing.md`.
- **`documentation/hpc.md`** — minor updates.
- **`documentation/xai.md`** — substantial expansion (+140 lines).
- **`eval/compare_results.ipynb`** — evaluation notebook updated.

### Fixed

- **TabPFN checkpoint deserialization** was entirely broken: `reintervention.py:load_checkpoint` only restored `ClassicalBranchWrapper` instances; `TabPFNBranchWrapper` was never restored from `classical_baseline_bytes`. The branch stayed in the unfitted zero-output state after loading a Stage A checkpoint into Stage B.

### Tests

- `test_tabpfn_backbone_has_non_differentiable_marker` renamed → **`test_tabpfn_backbone_is_differentiable`**: asserts `_is_non_differentiable is False` and `_manages_own_grad is True`.
- `test_tabpfn_is_classical_backbone` renamed → **`test_tabpfn_is_not_classical_backbone`**: asserts `not is_classical_backbone(m)`.

---

## [2026-06-08] — TabPFN single-checkpoint fix; median imputation; XGBoost NaN passthrough

### Added

- **`TabPFNBranchWrapper.model_dir`** — new `__init__` parameter. Globs `tabpfn-v2.5-classifier*.ckpt` from a local directory and passes the canonical **`_default.ckpt`** (matching `ModelSource.get_classifier_v2_5().default_filename`) as a single path string to `FinetunedTabPFNClassifier`, bypassing HuggingFace and `TabPFNLicenseError`. Weights at `/deepstore/datasets/mia/AI4EVAR/checkpoints/tabpfn/` (9 v2.5 checkpoints). Configured via `backbone_cfg.model_dir` in `experiments/v1_no_complications_tabpfn.yaml`. **Why single checkpoint**: passing a list triggers multi-model mode in `TabPFNClassifier`; `FinetunedTabPFNClassifier._fit` then calls `self.finetuned_estimator_.model_.to(device)` which raises `ValueError: The model_ property is not supported when multiple models are used`. `n_estimators_final_inference=4` still ensembles via different preprocessing seeds over the one fine-tuned model.
- **`compute_clin_stats`** now returns a third value — per-feature **median** (shape `(n_features,)`, float32). All-NaN features fall back to the mean. Used to compute `clin_fill = (median − mean) / std` (z-scored median), the new NaN fill value for `NormalizeClinicald`.
- **`NormalizeClinicald.fill_value`** — optional per-feature `np.ndarray` (in z-space). When provided, missing values are replaced by the z-scored median instead of 0 (mean). Falls back to `nan_to_num(..., nan=0.0)` when `None` for backward compatibility. Wired through `get_train/val_runtime_transforms` and `build_loaders` via a new `clin_fill` parameter.
- **9 new tests** in `tests/test_loaders.py`: `test_compute_clin_stats_returns_median`, `test_compute_clin_stats_all_nan_median_falls_back_to_mean`, `test_normalize_clinical_fill_value_used_for_nan`, `test_normalize_clinical_no_fill_defaults_to_zero`, `test_normalize_clinical_fill_value_only_replaces_nan`.
- **3 new tests** in `tests/test_baselines.py`: `test_xgboost_impute_preserves_nans`, `test_non_xgboost_impute_fills_nans_with_median`, `test_fit_classical_backbone_passes_nans_to_branch`.

### Fixed

- **`TabularBaseline._impute`** — XGBoost now returns the input array unchanged (NaNs preserved) so the model's native learned-direction split logic fires instead of a fixed median fill. All other backends (LR, RF, GBM) continue to receive median imputation.
- **`fit_classical_backbone`** — removed the `nan→0` pre-fill that silently overwrote NaNs with zero (mean in z-space) *before* `TabularBaseline._impute()` was called, making the median imputation in `_impute()` dead code. Classical backbones now receive z-scored features with NaNs intact, and each model's own imputation strategy fires correctly.

### Changed

- **`TabPFNBranchWrapper`** — `n_estimators_finetune` 2→1 and `n_estimators_validation` 2→1. Required to keep the fine-tuning estimator in single-model mode, consistent with passing a single checkpoint path.
- **`experiments/v1_xgboost.yaml`** — `checkpoint_dir` and `run_name` bumped to `_08062026`; re-submitted as SLURM job **510615** with corrected NaN handling.
- **`experiments/v1_logistic_regression.yaml`** — `checkpoint_dir` and `run_name` bumped to `_08062026`; re-submitted as SLURM job **510616** with corrected median imputation.
- **`experiments/v1_no_complications_tabpfn.yaml`** — `checkpoint_dir` and `run_name` bumped to `_08062026`; `backbone_cfg.model_dir` added; re-submitted as SLURM job **511368** loading weights from deepstore.

---

## [2026-06-05] — Add SAINT + TabPFN tabular backbones; feature gate exclusions; tests + docs

### Added

- **`src/models/saint.py`** — `SAINTEmbed`: SAINT tabular backbone from [somepago/saint](https://github.com/somepago/saint). Two styles: `col` (~66k params, column/feature attention only) and `colrow` (~850k params, column + intersample attention). Row FF uses a bottleneck MLP to avoid the upstream GEGLU×4 param explosion at large feature counts. Set via `backbone_cfg.style` in YAML.
- **`src/models/tabpfn.py`** — `TabPFNBranchWrapper`: TabPFN pretrained in-context transformer (Prior-Labs, Nature 2024) as a proper sibling module to `tabm.py` and `saint.py`. Fitted once via `FinetunedTabPFNClassifier` (lr=1e-5, 20 epochs, early stop) before Stage A, then frozen. Projection: **`Linear(n_estimators*2 → 32)`** using `predict_raw_logits()` — per-estimator raw logits `(E, B, C)` transposed to `(B, E*C)` so the 4-estimator diversity is preserved rather than averaged away. Width auto-computed from `n_estimators_final_inference` in `backbone_cfg`. Sets `_is_non_differentiable = True`.
- **`tests/test_tabpfn.py`** — 11 tests covering projection width = `n_estimators*2`, `n_estimators` stored, zero-before-fit guard, full model integration.
- **19 new tests in `tests/test_models.py`** covering SAINT for both `col` and `colrow` styles: output shape, finiteness, gradient flow, full `ReinterventionRiskModel` integration. Total suite: **389 tests**.
- **`experiments/v1_no_complications_saint.yaml`** — SAINT config; `style: col`, `dim=16, depth=2, heads=4`. SLURM job **509231**.
- **`experiments/v1_no_complications_tabpfn.yaml`** — TabPFN config; `n_estimators_final_inference: 4` (projection input = 8).
- **`tabpfn==8.0.6`** added to `requirements.txt` and `requirements-ci.txt`.
- **`eval/compare_results.ipynb` §7–9** — AUROC Forest Plot (95% bootstrap CI), Cross-model AUROC Matrix (Δ train−test overfitting flag), Pairwise Bootstrap Significance table. Full *What/How/How to read/Why* documentation added to all 9 section headers.

### Fixed

- **`torch.cuda.amp.autocast()` → `torch.amp.autocast('cuda')`** in `src/training/trainer.py` — silences PyTorch 2.x deprecation warning.

### Changed

- **`src/models/baselines.py`** — `CLASSICAL_BACKBONES = AVAILABLE = ["lr","rf","gbm","xgboost"]` (sklearn only). `ClassicalBranchWrapper` gains `_is_non_differentiable = True`. TabPFN was never a baseline and lives in its own module.
- **`src/models/reintervention.py`** — `is_classical_backbone()` uses `getattr(branch, "_is_non_differentiable", False)` marker instead of `isinstance` check; extensible to any future frozen backbone. `_build_clinical_branch` handles `"tabpfn"` in its own branch parallel to `"tabm"` and `"saint"`.
- **`src/training/trainer.py`** — `fit_classical_backbone()` guard uses `is_classical_backbone()` instead of `isinstance(ClassicalBranchWrapper)`; works for sklearn and TabPFN.
- **`tests/test_baselines.py`** — TabPFN tests moved to `test_tabpfn.py`.
- **`_EXCLUDE_FEATURES`** in `src/data/loaders.py`: added `"Site"` and `"CreatedInSite"` — both encode acquisition site and would introduce bias.
- **`filter_features_by_missingness`** in `src/data/loaders.py`: prints `[feature filter]` summary with kept/dropped counts and per-feature missingness to SLURM log.
- **`experiments/v1_no_complications.yaml`**: checkpoint dir suffix `MLP_03062026` → `MLP_04062026`.

---

## [2026-06-04] — Classical LR backbones; train metrics; multi-split eval; eval notebook; XAI fix; config renames

### Added

- **Classical ML tabular backbones** — `src/models/baselines.py` adds `TabularBaseline` (standalone sklearn Pipeline wrapper for LR / RF / GBM / XGBoost) and `ClassicalBranchWrapper(nn.Module)` (fits a `TabularBaseline` and projects its scalar `predict_proba` output to 32-dim via a learnable `Linear(1→32)`, enabling gradient flow into the FiLM fusion and visual branch). Set `clinical_backbone: lr | rf | gbm | xgboost` in any experiment YAML to use — the visual branch and all downstream fusion parameters continue to train normally via backprop.
- **`fit_classical_backbone()`** in `src/training/trainer.py` — fits the sklearn model once on z-normalised clinical + raw AAA features (exact same distribution as DataLoader inference) before Stage A begins. Called automatically when a classical backbone is detected.
- **`is_classical_backbone(model)`** in `src/models/reintervention.py` — helper that returns `True` when the tabular branch is a `ClassicalBranchWrapper`; used to skip the feature gate and gate L1 throughout training.
- **`build_tabular_arrays()`** in `src/data/loaders.py` — extracts `(X, y, pids)` as raw numpy arrays (NaN preserved, no MONAI transforms) for use by the classical backbone fitting step.
- **4 new experiment configs**: `v1_logistic_regression.yaml`, `v1_random_forest.yaml`, `v1_gradient_boosting.yaml`, `v1_xgboost.yaml` — full pipeline configs with the same 2-stage structure as the DL experiments. Submit exactly as with MLP/TabM: `sbatch scripts/submit_slurm.sh train --config experiments/v1_random_forest.yaml`.
- **`backbone_cfg:` config key** — replaces the old `tabm:` key for all backbone types. All 6 experiment YAMLs now use `backbone_cfg:` for hyperparameter overrides forwarded to whichever backbone is selected. Backward-compatible: the trainer and runner still fall back to `tabm:` when `backbone_cfg:` is absent.
- **Train-set metrics logged every epoch** — `validate()` now accepts a `split` parameter; the training loop calls it twice per epoch (split="val" and split="train"), logging all metrics (`{split}/reint_auc`, `{split}/reint_f1`, etc.) and prediction tables to W&B and console under the correct prefix. Epoch summary now shows `train_loss`, `val_loss`, `train_auc`, `val_auc`, and `val_f{β}`.
- **Train prediction tables** — `log_val_table()` also accepts `split`; logged under `eval/{split}_full_matrix_stage_{st}` every 3 epochs for both val and train.
- **Per-stage ROC curves in W&B** (train + val) — at the end of each stage the trainer reloads the best checkpoint and logs `roc/val_stage_{st}` and `roc/train_stage_{st}` as W&B images.
- **`roc_curve.png`** saved to output directory by `run_test()` for every evaluation run.
- **`filter_features_by_missingness()`** in `src/data/loaders.py` — drops clinical features missing in ≥ `max_missing_pct` of training patients before stats are computed. 198/240 features (82%) are ≥50% missing; after filtering the input reduces to ~42 informative features. Computed on training split only (no leakage); `feature_names` list saved in checkpoint.
- **`max_missing_pct: 0.5`** config key under `training:` in both DL experiment YAMLs.
- **`xgboost==3.2.0`** added to `requirements.txt` and `requirements-ci.txt`.
- **41 tests in `tests/test_baselines.py`** covering `TabularBaseline` (shape, NaN handling, save/load, predict-before-fit guard), `ClassicalBranchWrapper` (zeros before fit, forward shape/no-NaN after fit, gradients through projection layer, serialize/deserialize), `is_classical_backbone`, and `fit_classical_backbone` integration. Total suite: 353 tests.
- **`eval/compare_results.ipynb`** — new Jupyter notebook for interactive results comparison. Auto-detects both the new (`<run>/<split>/<stage>/`) and legacy flat result layouts. Features: dark-themed summary table with pastel AUC/F1/sensitivity gradients and gold best-value indicators; joined ROC curves with per-curve fill and best-operating-point markers; per-metric bar chart. Grouping controlled by `ROC_GROUP_BY` (`"run"` / `"stage"` / `"split"` / `"run+stage"`) — a single config change switches between comparing methods, stages, or train/val/test splits.

### Fixed

- **`_clinical_importance` level assignments were inverted** in `src/xai/explainer.py`. The old code computed `pct = (grads >= s).mean()` and labelled HIGH if `pct > 0.9` — but that means only 10% of features have a lower gradient, i.e. this is near-minimum-importance. Fixed by normalising gradients to [0, 1] and applying HIGH > 0.7, MED > 0.3 (same approach as `_aaa_importance`).
- **Removed unused `threshold_recall_target` / `val/reint_recall_thresh`** dead code. The recall-targeting threshold was computed and logged but never stored in the checkpoint or used at inference. Removed from `validate()` signature, `run_pipeline()`, W&B metrics, and both DL YAMLs. The F-beta-optimal threshold (`recall_beta: 2.0`) remains the sole threshold end-to-end.

### Changed

- **`validate()` signature**: added `split: str = "val"` parameter; all metric keys now use `f"{split}/..."` prefix; return value extended to 3-tuple `(loss, metrics, (probs, gt))` to expose raw arrays for ROC plotting. Default `split="val"` preserves backward compatibility.
- **`log_val_table()`**: added `split` parameter; W&B key changed from `eval/full_matrix_stage_{st}` to `eval/{split}_full_matrix_stage_{st}`.
- **`set_stage_a/b()` and `build_optimizer()`** in `stages.py`: `feature_gate.requires_grad` and the `feature_gate` param group are skipped when `is_classical_backbone()` is True — the gate has no gradient path through a non-differentiable branch.
- **Gate L1 regularisation** skipped for classical backbones in the training loop (same reason).
- **`ReinterventionRiskModel.forward()`**: classical backbones receive raw (un-gated) features; DL backbones continue to use `clin * sigmoid(feature_gate)`.
- **`load_checkpoint()`**: restores the serialized sklearn model from `classical_baseline_bytes` key when present.
- **Checkpoint save dict** for classical backbones includes `classical_baseline_bytes` (joblib-serialised sklearn pipeline).
- **`--split` in `scripts/test.py`** now accepts `all` in addition to `train / val / test`. `--split all` runs all three splits across all discovered stage checkpoints. Output path is `output_dir/split/stage` (was `output_dir/stage/split`). `--checkpoint` is now optional: omitting it auto-discovers all `best_stage_*.pt` files in `cfg.paths.checkpoint_dir`.
- **`run_test()` header** prints `{split.upper()} SET METRICS` (was hardcoded `TEST SET METRICS`).
- **Temperature calibration** applied in `runner.py` at inference time: `prob = sigmoid(logit / temperature)` using the `temperature` key saved in the checkpoint.
- **`gate_l1_weight` 0.0005 → 0.0001** in both DL experiment YAMLs. With the missingness filter reducing gate input from 240 to ~42 features, lower L1 allows selective sparsity rather than blanket suppression.
- **`v1_no_complications.yaml`**: `clinical_backbone` tabm → mlp; `backbone_cfg:` key; checkpoint dir updated to `…_MLP_…`.
- **`v1_with_complications.yaml`**: `clinical_backbone` mlp → tabm; `backbone_cfg:` key; checkpoint dir updated to `…_TabM_…`.
- **`scripts/submit_slurm.sh` test usage comments** corrected: removed invalid `--gradcam` flag (GradCAM is on by default; use `--no-gradcam` to disable), replaced with `--split all` for the multi-stage run and `--split test` for the single-checkpoint run; added a third `--no-gradcam` example for fast metrics-only runs.

---

## [2026-06-03] — Augmentation pipeline; sentinel fix; TabM sizing; DataLoader OOM fix; crop script

### Added

- **Expanded training augmentation pipeline** in `get_train_runtime_transforms` (`src/data/loaders.py`) — 12 transforms total, up from 3. Augmentation now fires on ≈98% of samples, effectively presenting a unique view of every training case each epoch.
  - **Spatial transforms** (applied jointly to `ct`, `lbl`, and `dst` before `BuildCTAd`, so all three volumes remain geometrically consistent): `RandFlipd` on axes 0/1/2 (p=0.5/0.5/0.4), `RandRotate90d` in the axial plane (p=0.5, max_k=3), `RandAffined` (±10° rotation all axes, ±10/10/5-voxel translation, p=0.3), `RandZoomd` (85–115% scale, `keep_size=True`, p=0.3). Applying spatial transforms to all three volumes together prevents the lumen mask and distance map from falling out of alignment with the CT.
  - **Intensity transforms** (applied to `ct` only — `lbl` is a label map and `dst` a distance field, neither should be blurred or shifted): `RandAdjustContrastd` (γ=0.7–1.3, p=0.3; simulates scanner-to-scanner contrast variation), `RandShiftIntensityd` (±0.1 additive, p=0.3; HU calibration offset), `RandScaleIntensityd` (±15%, p=0.5; multiplicative scaling), `RandBiasFieldd` (coeff 0–0.3, p=0.2), `RandGaussianNoised` (σ=0.05, p=0.3), `RandGaussianSmoothd` (σ_z=0.25–0.5, half of σ_x/y, reflecting the 3 mm axial slice spacing, p=0.2).
  - Note: spatial augmentations (zoom, affine) do not recompute the pre-stored `aaa_metrics` scalars — those remain the clinically authoritative values, and the model learns robustness from the minor image/scalar mismatch.
- **35 new tests** in `tests/test_loaders.py` (total test count 274 → 312):
  - **Augmentation pipeline (20 tests):** composition (all 10 augment classes present, 3 flips, val has none), key coverage (spatial ops on `lbl`+`dst`, intensity ops on `ct` only), ordering (all augments precede `BuildCTAd`), flip correctness (each axis reverses the right spatial dim; `ct` and `lbl` flipped identically; values preserved), intensity isolation (6 transforms each leave `lbl`+`dst` bit-identical), shape preservation (zoom/affine keep `(D,H,W)`; zoom nearest-neighbour keeps `lbl` in `{0,1,2}`), stochasticity (two passes on the same input differ).
  - **Sentinel and missing-value handling (15 tests):** 9999/999/99999 and their string forms become NaN; 9998 passes through; date strings become NaN; `PatientNr` absent from cleaned dict and from `feature_names`; `compute_clin_stats` excludes NaN from mean/std; all-NaN features get neutral stats (mean=0, std≈1); `NormalizeClinicald` imputes NaN to z=0 after normalising; sentinel patients get z=0 for sentinel feature; no NaN survives into model; `build_data_list` preserves NaN for downstream imputation.

### Fixed

- **Complication label mismatches in `03_data_matching_v1.py`** (`preprocessing/case_matching_v1/`). Three bugs fixed:
  - **Regex anchors missing from `COMP_PATTERNS`**: patterns like `answer1` matched `answer10`, `answer11`, etc., causing wrong endoleak-type and migration/infection labels. Added `$` anchors to all answer-indexed patterns (e.g. `answer1$`, `answer(1|2)$`).
  - **Adverse-event (AE) columns ignored**: complication timing was derived only from `FU`-prefixed columns (`FU1_...`, `FU2_...`). Complications documented under `AE{N}_AE_AE_EDate` columns (adverse-event reports, a distinct pathway in RADAR) were silently discarded. Added `_extract_ae_number` and `_find_ae_date` helpers; `extract_ground_truth` now collects both FU and AE candidates and picks the earliest by `days_from_op`.
  - **Analysis report print** updated to show the source (`FU{N}` / `AE {N}`) and the triggering field name for each detected complication, replacing the previous `(no FU-prefixed column found)` fallback with the more accurate `(no complication event found)`.

- **Checkpoint threshold now uses F-beta-optimal value instead of recall-targeting value** (`src/training/trainer.py`). Previously the saved `"threshold"` key was `val/reint_recall_thresh` — the largest threshold still achieving ≥80% recall on val. This is degenerate: a model predicting all positives satisfies the criterion trivially, producing high recall and near-zero precision at test time (observed with TabM: TP=17, FP=71, accuracy=0.34). Changed to `val/reint_best_thresh` — the threshold that maximises F-beta (F1 with default `recall_beta=1.0`) on the val PR curve. To weight recall over precision, set `recall_beta > 1` in config; the threshold still balances precision and recall rather than ignoring precision entirely. Checkpoint log line updated to show `thresh=X@F1` instead of `thresh=X@80%rec`. All existing checkpoints carry the old recall-targeting threshold and should be retrained.

- **DataLoader CPU RAM OOM** causing SLURM jobs to be killed around epoch 18 on 180 GB nodes (`srun: error: task 0: Killed`). Two compounding causes: (1) with `v1_with_complications`, 16 train + 16 val = 32 persistent workers each holding `prefetch_factor=2` batches of 128 MB 4-channel CTA tensors ≈ 64 GB in worker buffers alone; (2) Python copy-on-write degradation — forked worker processes accumulate RSS epoch-by-epoch as CPython reference-counting touches shared `CacheDataset` pages, causing each worker's resident set to grow toward the full cache size. Fix applied in `src/data/loaders.py`:
  - **`prefetch_factor` 2 → 1** for both train and val loaders — halves per-worker buffer RAM immediately.
  - **Val loader now uses `max(2, num_workers // 4)` workers** — val is 100%-cached and runs once per epoch; it does not need a large worker pool.
  - **`num_workers` reduced** in experiment configs: `v1_no_complications` 8 → 4; `v1_with_complications` 16 → 6. Peak worker-buffer RAM drops from ~64 GB to ~10 GB.
- **Clinical branch near-zero modality importance** (0–1% across test patients) caused by RADAR sentinel values (999, 9999, 99999) being treated as real clinical measurements. Root cause: 155 of 241 features are >80% missing and use these sentinels for "not measured"; including them in `compute_clin_stats` inflated the mean to ~8000 for affected features (e.g. `Bl_Bl_PH_FEV` with 80% 9999s: mean ≈ 8100 instead of ≈ 80). After z-normalisation, real patient values became z ≈ −30 to −50 (extreme outliers) while sentinel patients got z ≈ 0 (looked normal). The L1 feature gate correctly learned to zero out all contaminated features — but that meant zeroing almost everything. Fixed in `_clean_clinical_features`, `compute_clin_stats`, `build_data_list`, and `NormalizeClinicald`:
  - **`_SENTINEL_VALUES = frozenset({999.0, 9999.0, 99999.0})`** and **`_EXCLUDE_FEATURES = frozenset({"PatientNr"})`** — named constants, single source of truth.
  - **`_clean_clinical_features`**: sentinel values → `np.nan`; non-numeric strings (date fields like `"2010-12-07"`) → `np.nan` (previously silently became `0.0`); `None`/empty → `np.nan`; `PatientNr` (database ID up to ~1e8, no clinical meaning) excluded entirely.
  - **`compute_clin_stats`**: `matrix.mean()` / `matrix.std()` → `np.nanmean()` / `np.nanstd()` so sentinels that became NaN are excluded. Features that are 100% missing get `mean=0, std=1` (neutral: normalise to z=0 for every patient, gate zeros them cheaply).
  - **`build_data_list`**: `nan_to_num` removed from `clin` array — NaN propagates to `NormalizeClinicald` so imputation happens in z-space, not raw space.
  - **`NormalizeClinicald.__call__`**: `torch.nan_to_num(normalised, nan=0.0)` after normalising — missing/sentinel patients get z=0 (training mean), the most agnostic imputation.

### Changed

- **TabM `d_block` 128 → 64; `d_embedding` 16 → 8** in both experiment YAMLs. At `d_block=128` the TabM clinical branch had 671K parameters = 10,646 per training positive — severely over-parameterised for a 63-positive dataset. Reducing to `d_block=64` and `d_embedding=8` brings this to ~282K = ~4,474 per positive, a more reasonable ratio while retaining the ensemble benefit of `k=16` members.
- **`v1_with_complications.yaml`**: `clinical_backbone` set to `tabm` and `checkpoint_dir` updated to `v1_with_complications_TabM_03062026` for the next TabM training run with the corrected sentinel handling and reduced `d_block`.
- **`v1_no_complications.yaml`**: `clinical_backbone` set to `tabm`; `checkpoint_dir` updated to `v1_no_complications_TabM_03062026`.
- **Crop script (`06_crop_full_body_scans.py`)**: `metricsTr/*.json` now copied to `metricsTr{suffix}/` when `--suffix` is set (previously absent, making suffixed directories unusable for training); dry-run crop counter fixed (`[DRY ]` tags now counted as would-crop, not skipped); non-CT copy loop prints flush=True with a progress counter every 50 files so SLURM logs confirm progress.

---

## [2026-06-02] — Add TabM backbone; merge AAA scalars; fix GradCAM planes + training regression

### Added

- **`pytest.ini`** (new file). Suppresses two unavoidable third-party import warnings: `FutureWarning: The cuda.cudart module is deprecated` (triggered by MONAI/CUDA on import) and `DeprecationWarning: builtin type … has no __module__ attribute` (SWIG bindings from VTK/ITK transitive dependencies). Neither is fixable in project code; both are silenced with targeted `filterwarnings` patterns.
- **`ReinterventionXAI._build_views` and `_orient` static methods** (`src/xai/explainer.py`). Extracted the duplicated view-construction code from `save_gradcam_pngs` and `get_gradcam_wandb_images` into a single helper. `_orient` applies `np.flipud(arr.T)` to put anatomical "up" at screen top for all three planes (superior at top for sagittal/coronal, anterior at top for axial).

- **`src/models/tabm.py`** — new `TabMEmbed` module wrapping the official `tabm` package (Gorishniy et al., 2024, "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"). Trains an ensemble of *k* MLPs with shared weights and per-member rank-1 BatchEnsemble adapters. Returns `(B, k, 32)`; reduces over the ensemble axis via mean to satisfy the `(B, 32)` branch contract. No reimplementation — uses `TabM.make(n_num_features=..., cat_cardinalities=[], d_out=32, **tabm_cfg)` from the library directly.
- **`PeriodicEmbeddings` numerical feature encoding** (default `num_embeddings: periodic`). Before the MLP ensemble each scalar feature is projected to a `d_embedding`-dim sinusoidal representation `[sin(f₁x), cos(f₁x), …]` via `rtdl_num_embeddings.PeriodicEmbeddings`. Uses `lite=True` (shared frequencies across all 744 features, 48 learnable params) instead of per-feature frequencies (~35 K params) — the right choice for 377 training samples. Disable with `model.tabm.num_embeddings: none`.
- **`clinical_backbone` config key** (`model.clinical_backbone: mlp | tabm`, default `mlp`). Selects the clinical encoder at construction time with no code changes. Switch via YAML or runtime override: `--set model.clinical_backbone=tabm`.
- **`tabm:` block** in both experiment YAMLs (`model.tabm: {k, n_blocks, d_block, dropout, num_embeddings, d_embedding}`). Only consumed when `clinical_backbone: tabm`; has no effect on the MLP path.
- **`clinical_backbone` and `tabm_cfg` saved to checkpoints** in `trainer.py` and read back in `runner.py` (same fallback pattern as `complications`). Ensures `test.py` rebuilds the exact architecture that was trained.
- **`tabm==0.0.3` and `rtdl_num_embeddings==0.0.12`** added to `requirements.txt` and `requirements-ci.txt`.
- **20 new tests**: `tests/test_tabm.py` (13 tests — output shape, determinism, PeriodicEmbeddings and none paths) and `tests/test_models.py` TabM section (7 tests — model builds, backbone stored, forward shapes, clinical branch output dim, unknown backbone raises). Total test count: 254 → 274.
- **`_build_clinical_branch` static method** on `ReinterventionRiskModel`. Centralises MLP / TabM construction; raises `ValueError` for unknown backbone names.

### Fixed

- **Training regression (always-predict-1 + AUC collapse)** introduced in the 01062026 commit by reverting four destabilising changes while preserving the clean loss simplification:
  - **`pos_weight`: 2.0 → 1.0** in both experiment configs. `WeightedRandomSampler` already balances batches 50/50; `pos_weight=2.0` on top was double-correcting class imbalance, driving the model to predict positive for every sample (TNR dropped from 0.51 to 0.01).
  - **`focal_gamma`: 3.0 → 2.0**. γ=3 compounded the positive bias by further down-weighting confident-correct negatives under an already 50/50 sampler.
  - **`gate_l1_weight`: 0.005 → 0.0005**. 10× sparsity was zeroing the clinical gate entirely, collapsing the main non-visual discriminator and tanking AUC from 0.665 to 0.556.
  - **Stage B: layer4 conv weights re-frozen; BN-only adaptation restored**. Fine-tuning a full 3D ResNet conv block on 63 training positives corrupted Merlin's pretrained features. BN-only adaptation (~50 K params) recalibrates scanner-domain statistics safely.
  - **`clinical_branch` restored to MLP** (`Linear(744→128)→LN→ReLU→Dropout(0.3)→Linear(128→32)→Dropout(0.2)`). The linear probe combined with 10× L1 gutted clinical signal from two directions simultaneously.
- **GradCAM plane labels and display orientation corrected** (`src/xai/explainer.py`). In RAS-oriented data (axis 0 = R, 1 = A, 2 = S), the old `views` dict had all three labels wrong: `ct[D//2]` slices along R → **sagittal** (was labeled "axial"); `ct[:, H//2]` slices along A → **coronal** (was labeled "sagittal"); `ct[:, :, W//2]` slices along S → **axial** (was labeled "coronal"). Fixed across both `save_gradcam_pngs` (PNG output) and `get_gradcam_wandb_images` (W&B logging). Each slice also receives `_orient` (`np.flipud(arr.T)`) so superior is at screen top for sagittal/coronal and anterior is at screen top for axial.
- **`Orientationd(labels=None)`** in `get_cached_transforms` (`src/data/loaders.py`). Explicitly opts into the new MONAI default rather than relying on the deprecated implicit value, silencing `FutureWarning: … labels … was changed in version None`.
- **`build_data_list` path resolution** (`src/data/loaders.py`). Paths are now read from the config JSON (`pre_op["nifti"]`, `pre_op["label"]`, `pre_op["dist_map"]`, `pre_op["metrics"]`) instead of being reconstructed from `base_dir`. Fixes a split-of-zero bug where stale v1 paths stored in a v2 config caused all 577 patients to fail the NIfTI existence check and be skipped.
- **`_load_aaa_metrics` accepts `str | None`** (was typed as `Path`). Calling `.exists()` on a string path or `None` raised `AttributeError`; now wraps in `Path()` and guards `None` explicitly.
- **`pre_op` null guard in `build_data_list`**: `(pdata["scans"].get("pre_op")) or {}` prevents `AttributeError` when a JSON-null `pre_op` entry is encountered.

### Changed

- **`aaa_metrics_branch` removed; AAA scalars merged into the tabular branch** (`src/models/reintervention.py`). The dedicated `Linear(5→32)→ReLU→Linear(32→32)→ReLU` branch is gone. In `forward`, the 5 AAA geometry scalars are now `cat`-appended to the gated clinical vector *after* the sigmoid gate and passed together into `clinical_branch`. **Why**: the old design gave 5 scalars a 32-dim fusion slot equal to 744 clinical features (disproportionate vote), and `Linear(5→32→32)` (~1,200 params) was heavily overparameterised for 5 inputs. Appending after the gate means scalars bypass L1 sparsity pressure (they are always informative) while sharing the MLP/TabM encoder fairly.
- **Fusion dimension 320 → 288** (`_fusion_dim`). Two branches (256-dim visual + 32-dim tabular) instead of three (256 + 32 + 32). `condition_net`, `fusion_norm`, and both output heads updated accordingly.
- **`clinical_branch` input dim extended**: `_build_clinical_branch` is now called with `clinical_dim + aaa_dim` (749 for RADAR). MLP first layer: `Linear(744→128)` → `Linear(749→128)`. TabM: `n_num_features` = 749. Feature gate stays shape `(clinical_dim,)` — it only covers clinical features.
- **`set_stage_a` / `set_stage_b`** in `stages.py`: removed the `aaa_metrics_branch` freeze/unfreeze blocks (attribute no longer exists on the model).
- **`build_optimizer` param groups reduced**: Stage A 6 → 5 (no-comp), 7 → 6 (with-comp); Stage B 9 → 8. Removed the `"aaa"` param group in both stages.
- **`print_model_param_overview`**: removed `aaa_metrics_branch` row.
- **Architecture docs comprehensively updated** (`documentation/architecture.md`): diagram now shows two-branch fusion at 288-dim; new "Tabular Branch" section explains gate-then-append design; "Why merge?" rationale added to key-design-choices table; all head/FiLM dim references corrected 320 → 288.
- **Actual split stats updated** across both experiment configs and comments: train N=377, 63 positives, 314 negatives (raw `pos_weight` ratio = 4.98 — must not be used with WRS). Previous comments referenced "~37 training positives" from an outdated estimate.
- **Checkpoint directories renamed**: `v1_no_complications` → `v1_no_complications_02062026`, `v1_with_complications` → `v1_with_complications_02062026`, to keep new-split runs cleanly separated from prior checkpoints.
- **`cache_dir` removed** from paths config (both YAMLs and `configuration.md`). Was already unused in `trainer.py` after a previous cleanup; stale key removed.
- **`p_cta` documentation**: corrected default to `0.0` in `configuration.md` (was showing old `0.10`).
- **Stage B optimizer**: removed `i3resnet_layer4` param group; Stage B groups 10 → 9 (no-comp), 11 → 10 (with-comp). Test `test_build_optimizer_stage_b_group_count` updated accordingly.

### Tests

- **Fusion dim assertions updated 320 → 288**: `test_return_fused_no_comp`, `test_return_fused_with_comp`, and all three `test_film_*` clamp tests.
- **`test_tabm_clinical_branch_output_dim`**: updated to cat gated clinical + aaa before calling `clinical_branch` directly (branch now expects 749-dim input).
- **`test_set_stage_a_trains_aaa_branch` → `test_set_stage_a_no_aaa_branch`**: now asserts that `aaa_metrics_branch` does *not* exist on the model.
- **Optimizer group count tests**: Stage A 6 → 5, Stage A with-comp 7 → 6, Stage B 9 → 8.

---

## [2026-06-01] — Linear probe; plain focal BCE; layer4 fine-tune; v3 hyperparams

### Changed

- **`clinical_branch` architecture: 2-layer MLP → linear probe** (`nn.Linear(clinical_dim, 32)`). With ~37 training positives and 241 clinical features a hidden MLP memorises training cases; a single linear projection over gated features cannot. Removes `LayerNorm`, two `Dropout` layers, and the 128-dim hidden layer entirely.
- **`gate_l1_weight`: 0.0005 → 0.005** (10×) in all experiment configs. Forces the learned feature gate to retain only ~20–30 clinical features, preventing spurious correlations from driving the high-confidence false positives seen in v1 evaluation.
- **`focal_gamma`: 2.0 → 3.0** in all experiment configs. Sharpens the gradient on borderline (mid-probability) examples in the 0.2–0.5 range where FPs and TPs were overlapping.
- **`pos_weight`: 1.0 → 2.0** in all experiment configs. Adds asymmetric gradient emphasis on false negatives even with the 50/50 `WeightedRandomSampler`, pushing true-positive probabilities higher relative to true-negative probabilities.
- **`p_cta`: 0.10 → 0.0** in all experiment configs. CTA branch confirmed useful by ablation study (`v1_no_complications_ablation_no_cta` achieved lower AUC); dropping it 10% of the time during training weakened the most informative branch.
- **`reintervention_head` and `complications_head` output width**: `2 → 1` and `n_comp × 2 → n_comp`. Removes the log-variance output from both heads.
- **Loss functions: heteroscedastic → plain focal BCE**. Removed `_heteroscedastic` helper from `losses.py`. Both `ReinterventionLoss` and `MultiTaskReinterventionLoss` now compute `_focal_bce_per_sample(...).mean()` directly. The heteroscedastic formulation was converging to `logvar≈0` (KL prior minimum), making it a mathematical no-op and wasting model capacity.
- **`ReinterventionLoss.forward` signature**: removed `reint_logvar` parameter. New signature: `forward(reint_logit, reint_gt)`.
- **`MultiTaskReinterventionLoss.forward` signature**: removed `reint_logvar` and `comp_logvar` parameters. New signature: `forward(reint_logit, comp_logits, reint_gt, comp_gt)`.
- **`details_dict` from both losses**: removed `mean_uncertainty` key (was always `≈0` when `logvar→0`).
- **`ReinterventionRiskModel.forward` return tuple**: no-comp variant returns `(logit,)` instead of `(logit, logvar)`; with-comp variant returns `(logit, comp_logits)` instead of `(logit, logvar, comp_logits, comp_logvar)`.
- **Stage B visual backbone training**: `set_stage_b` now unfreezes all `layer4` parameters (conv + BN), not just BN. Merlin's `layer4` encodes general radiology features; task-specific fine-tuning lets it learn EVAR re-intervention patterns.
- **Stage B optimizer**: added `i3resnet_layer4` parameter group at LR 5e-6. Layer4 conv weights are updated at a conservative rate separate from BN adaptation (1e-5). Total Stage B param groups: 9 → 10 (no-comp), 10 → 11 (with-comp).
- **Confidence formula**: `1 / (1 + exp(0.5 × logvar))` → `2 × |prob − 0.5|` in `ReinterventionXAI.full_explain`. Self-calibrating: confidence is 0 when the model is maximally uncertain (prob ≈ 0.5) and 1 when it is certain (prob near 0 or 1). No post-hoc tuning required.
- **`_conf_level` thresholds**: HIGH `> 0.75` → `> 0.60`, MED `> 0.55` → `> 0.30`. Calibrated to the `2×|prob−0.5|` scale (HIGH ≡ prob outside [0.20, 0.80]; MED ≡ prob outside [0.35, 0.65]).
- **`validate()` metrics**: removed `val/mean_uncertainty` (was always ~0 with `logvar→0`).
- **`prediction.json`**: removed `reint_logvar` field from per-patient output.

### Removed

- **`_heteroscedastic` function** from `src/training/losses.py`. The KL-style symmetric prior `0.5*(exp(logvar)−1−logvar)` was correct in theory but ineffective in practice: with AUC 0.665 the model was globally uncertain and kept `logvar≈0` (the prior minimum), reducing the weighted loss to identical focal BCE while adding dead parameters.

---

## [2026-05-30] — Fix double pos_weight; recall-threshold checkpoint; CTA-ablation config

### Added

- **`experiments/v1_no_complications_ablation_no_cta.yaml`** — new ablation config that trains on clinical features + AAA geometry only (CTA branch permanently zeroed). Motivated by XAI analysis showing CTA modality importance of 0.002–0.20 across test patients, suggesting the frozen Merlin backbone may add noise rather than signal. Runs Stage A only (80 epochs, patience 20); `xai_gradcam: false` since GradCAM is meaningless with zeroed input.
- **`disable_cta` flag** (`training.disable_cta: true`) in experiment configs. When set, the CTA tensor is zeroed before every forward pass in training, validation, and test — not just during modality dropout. Supported in `trainer.py` (`run_pipeline`, `validate`) and `runner.py` (`run_test`).
- **`threshold_recall_target`** config key (default `0.80`). Checkpoints now save the largest val threshold achieving ≥`threshold_recall_target` recall, instead of the F2-optimal threshold. With only 9 val positives the F2-optimal threshold is volatile; a recall-targeting threshold is more stable and transfers better to test. The saved value is logged as `val/reint_recall_thresh` to W&B and printed in the checkpoint line as `thresh=X.XXX@80%rec`.

### Fixed

- **Double imbalance correction** in both experiment configs: `pos_weight: 4.58 → 1.0`. `WeightedRandomSampler` already creates ~50/50 balanced batches; applying `pos_weight=4.58` on top gave positives ~21× more loss emphasis than the data warrants. This inflated the model's output probabilities for training-set positive patterns, anchoring the val threshold at ~0.57 — which then failed catastrophically on test (recall 0.15 at that threshold vs 0.95 at threshold 0.25).

### Changed

- **`p_clinical: 0.30 → 0.15`** in both experiment configs. 30% clinical feature dropout is overly aggressive when the training set contains only ~37 positives; the model needs more stable clinical signal to learn reliable reintervention patterns.
- **`v1_with_complications` Stage A `patience: 12 → 20`**. The previous run hit early stopping at epoch 15 with val AUC 0.656 — well below the no-complications model's 0.807 at epoch 33. The multi-task complication head adds gradient noise and needs more epochs to converge.
- **`preprocessing/prepare_data.sh`** — active SLURM step switched from step 05 (nnUNet labels + metrics generation) to step 06 (full-body crop); log file names updated accordingly.

---

## [2026-05-29] — F2-score checkpoint; Merlin geometry; complication timing; nnUNet fixes; CI pipeline; crop script suffix output

### Added

- **`diam_mapping` field** in `ground_truth` output of `03_data_matching_v1.py`. Records which diameter-pairing strategy was used per patient: `CT-CT` (preferred) → `US-US` → cross-modality (`CT-US`, `US-CT`) → `GEN-*` fallback (generic `Bl_AAAdiam` column for older Rijnstate patients). A new **DIAMETER MAPPING STRATEGY** table appears in both the console report and the `_analysis.txt` file, with a legend explaining each code.
- **Complication timing in `ground_truth`** (`03_data_matching_v1.py`). For each of the 11 complication labels (endoleak types, occlusion, stenosis, kinking, fracture, migration, infection), a `comp_timing` dict is now stored: `{"first_fu": 3, "first_date": "2007-06-12", "days_from_op": 601}`. Determined by finding the earliest FU-prefixed flagged column and resolving its date from `FU{N}_Vas{N}_Fu_VDate_...`. The analysis report now prints per-patient timing lines instead of a flat patient-ID list for complication metrics.
- **Minimum physical-extent guarantee** in `06_crop_full_body_scans.py`. After the aortic crop, the script now expands the window symmetrically if the axial physical depth is below 480 mm (= 160 slices × 3 mm), ensuring `SpatialPadd` adds minimal zero-padding. Status line now also prints physical depth in mm.
- **9 new tests** covering `_make_weighted_sampler` (weight correctness, type, count), `get_cached_transforms` pipeline contents (`SpatialPadd` present, `Resized` absent), `val/reint_best_fbeta` key presence and range, and F-beta finite-output check.
- **`val/reint_best_fbeta` added to required-keys set** in `test_trainer.py` (`test_validate_no_comp_required_keys`).
- **GitHub Actions CI workflow** (`.github/workflows/ci.yml`). Runs the full 220-test CPU-only pytest suite on every push to `main`, `dev*`, and `feature*` branches and on PRs targeting `main`. On push to `main`, generates `coverage.json` and `tests.json` badge data and deploys them to a `gh-badges` branch via `JamesIves/github-pages-deploy-action`.
- **`requirements-ci.txt`** — stripped-down dependency list for CI: `torch` (CPU), `monai`, `scikit-learn`, `scipy`, `numpy`, `matplotlib`, `wandb`, `tqdm`, `einops`, `PyYAML`, `pytest`, `pytest-cov`. Excludes CUDA packages, `merlin-vlm`, `peft`, `bitsandbytes`, `nnunetv2`, VTK/PyVista, and other HPC-only dependencies.
- **Dynamic health badges** in `README.md`: CI pass/fail (GitHub Actions built-in), **tests** (`x/220 passed`), **coverage** (`x%`) via shields.io endpoint reading the `gh-badges` branch, **last commit** via shields.io GitHub API. Plus static **Python 3.11** and **Apache-2.0** badges.
- **`--suffix` argument** in `06_crop_full_body_scans.py`. Appending e.g. `--suffix _cropped` writes output to `imagesTr_cropped/`, `labelsTr_cropped/`, `distMapsTr_cropped/` instead of overwriting in-place, letting the pipeline produce a cropped derivative without destroying the source data.

### Changed

- **Checkpoint criterion: AUC → F2-score** (`trainer.py`). Model is now saved when `val/reint_best_fbeta` (β=2, recall-weighted) improves rather than AUC. Early-stopping patience and epoch log lines updated accordingly. Directly optimises for sensitivity over discrimination — missing a reintervention is clinically far worse than a false alarm.
- **`WeightedRandomSampler` added** (`loaders.py`). Train DataLoader now draws positives and negatives with equal expected frequency (weight = 1/N_class), producing ~50/50 batches. Replaces the previous `shuffle=True` with natural ~18/82 skew. `pos_weight` stays at 4.58 to encode residual clinical FN-penalty without double-counting the balance.
- **Image geometry aligned to Merlin's pretrained pipeline** (both experiment YAMLs, `loaders.py`):
  - `image_size` 128³ → `[224, 224, 160]`
  - `voxel_spacing` `[1.5, 1.5, 1.5]` → `[1.5, 1.5, 3]` mm
  - `hu_range` `[-200, 400]` → `[-1000, 1000]` HU
- **`Resized` → `SpatialPadd` + `CenterSpatialCropd`** (`loaders.py`). Replaces trilinear resize with Merlin's exact preprocessing: zero-pad volumes smaller than the target, then center-crop to exactly 224×224×160. Preserves voxel structure rather than stretching.
- **Regularisation loosened** (both experiment YAMLs, `reintervention.py`):
  - `gate_l1_weight` 0.002 → 0.0005 (previous value was zeroing too many of the 241 clinical features)
  - `label_smoothing` 0.05 → 0.02
  - Clinical branch Dropout 0.4/0.3 → 0.3/0.2
  - FiLM clamp 0.15 → 0.3 (doubles the modulation range; allows meaningful cross-modal fusion)
- **`comp_weight_stage_a`** 0.1 → 0.01 (`v1_with_complications.yaml`). Near-zero during Stage A so the reintervention head warms up cleanly before the 7-task complication gradients are introduced.
- **`crop_volume()`** (`06_crop_full_body_scans.py`): added optional `out_path` parameter — writes to that path when provided, otherwise overwrites in-place as before.
- **`process_scan()` status output** (`06_crop_full_body_scans.py`): single-line status replaced with a structured multi-line block showing input shape, output shape, axial extent in slices and mm, whether the 480 mm minimum-extent floor was enforced, and which output directories were written.
- **`--input-dir` injection guard** in `06_crop_full_body_scans.py` `__main__` block: changed from `len(sys.argv) == 1` to `"--input-dir" not in sys.argv` so that running with only `--suffix _cropped` still injects the default `INPUT_DIR` correctly.
- **`prepare_data.sh`**: updated the commented-out step-06 invocation to include explicit `--input-dir` and `--suffix _cropped` arguments, matching the new CLI interface.

### Fixed

- **nnUNet cwd crash** (`05_nnUNet_labels_metrics_generation.py`). nnUNet internally `chdir`s during inference; after the first scan's temp dir was deleted by `shutil.rmtree`, the next `multiprocessing.Pool` spawn called `os.getcwd()` on a non-existent path, raising `FileNotFoundError`. Fixed by calling `os.chdir(output_base)` before each `predict_from_files` call to keep the process cwd pinned to a stable directory.
- **nnUNet missing env-var warnings** (`05_nnUNet_labels_metrics_generation.py`). `nnunetv2` checks `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` at import time and prints loud warnings if unset. Fallback values are now set with `os.environ.setdefault` before the import, suppressing the noise without overriding real environment values.
- Removed stale misleading comment in `losses.py` that claimed `WeightedRandomSampler` was already in use (it was not).
- All documentation (`architecture.md`, `configuration.md`, `data.md`, `training.md`, `testing.md`) updated to reflect new geometry, regularisation values, FiLM clamp, checkpoint criterion, and WeightedRandomSampler.

---

## [2026-05-28] — Stronger regularisation; per-complication loss weights; preprocessing refinements

### Changed
- **Stronger regularisation** across both experiments:
  - `gate_l1_weight` 0.0001 → 0.002 (10× stronger clinical feature sparsity)
  - `label_smoothing` 0.02 → 0.05
  - Clinical branch Dropout 0.2/0.1 → 0.4/0.3
  - `weight_decay` 1e-4 → 5e-4
  - FiLM clamp 0.3 → 0.15 (tighter modulation)
- **Modality dropout in Stage A**: `p_clinical` 0.20 → 0.30; dropout now active during the tabular warm-up stage, not just Stage B
- **Slimmer `condition_net`**: hidden dim 512 → 256 (320→256→640; ~50% fewer params)
- **`v1_with_complications`**: per-complication `pos_weights` (N_neg/N_pos, capped at 20) and `comp_loss_weights` (scaled by √(N_pos/N_pos_max)) to balance rare complication tasks
- **`preprocessing/case_matching_v2`**: refactored `03_data_matching_v1.py` and `01_build_session_index.py` for improved robustness and clarity
- Minor fixes in `losses.py`, `trainer.py`, and `reintervention.py`

---

## [2026-05-23] — LoRA removal; BatchNorm adaptation

### Removed
- **LoRA / PEFT dependency** dropped entirely. LoRA only targeted `Linear` layers in Merlin's text encoder, leaving the I3ResNet image encoder (the branch we actually train) untouched — making it ineffective for visual domain adaptation.

### Added
- **BatchNorm adaptation** for I3ResNet `layer3` and `layer4` inside `encode_image`. Stage B now trains BN γ/β at `lr=1e-5` instead of LoRA weights, recalibrating feature statistics for scanner protocol differences (MST/UMCG/AUMC) without touching conv weights.
- New `06_crop_full_body_scans.py` preprocessing step in `case_matching_v2`.
- `07_generate_data_split.py` (renumbered from `06_`) in `case_matching_v2`.

### Changed
- `stages.py`: Stage B unfreezes BN params (strictly within `encode_image`) instead of LoRA params; optimizer param group renamed `lora` → `i3resnet_bn`.
- Experiment configs: removed `lora_r` key, updated stage-B descriptions.
- `prepare_data.sh` updated to reflect new pipeline step numbering.

---

## [2026-05-21] — v2 data preparation pipeline

### Added
- **`preprocessing/case_matching_v2/`** — full multi-site, multi-modality pipeline supporting pre- and post-operative data for SSL backbone training:
  - `00_dicom_to_nifti_renaming.py` — DICOM → NIfTI conversion handling two XNAT layouts (flat-session: MST/RNST; layered-session: AUMC/UMCG) and all imaging modalities (CT, US, XR, MR)
  - `01_build_session_index.py` — builds a structured session index across sites
  - `02_cfd_name_cleaning.py` — CFD filename normalisation
  - `03_data_matching_v2.py` — cross-site patient/session matching
  - `04_data_organizing.py` — output directory organisation
  - `05_nnUNet_labels_metrics_generation.py` — nnUNet segmentation + AAA geometry metrics
  - `06_generate_data_split.py` — stratified train/val/test split generation
- `preprocessing/download_data_xnat.py` — XNAT data download helper

### Removed
- Monolithic notebooks removed from `src/notebooks/` (`reintervention_no_physics_no_complications.py`, `reintervention_with_complications.py`, and their dated snapshots, `test_*.py` notebooks, `reint.sh`).

### Changed
- `MultiTaskReinterventionLoss` refinements in `losses.py`.
- `tests/test_models.py` updated for revised model signature.
- Documentation: `documentation/training.md` updated with pipeline notes.

---

## [2026-05-19] — Documentation suite; test suite expansion

### Added
- **Full documentation suite** under `documentation/`:
  - `architecture.md` — model architecture, branches, FiLM fusion, stage curriculum
  - `configuration.md` — YAML config reference
  - `data.md` — data layout, MONAI pipeline, transform details
  - `evaluation.md` — metrics, inference, GradCAM
  - `hpc.md` — SLURM job submission, HPC environment
  - `testing.md` — pytest structure, fixtures, coverage notes
  - `training.md` — training workflow, W&B logging, checkpointing
  - `wandb.md` — experiment tracking setup
  - `xai.md` — XAI methods and usage
- **Expanded test coverage**:
  - `tests/test_trainer.py` — covers `validate()`, full training loop integration
  - `tests/test_runner.py` — covers `run_test()`, per-patient output format
  - `tests/test_scripts.py` — smoke tests for `train.py` and `test.py` entry points
  - Additional `test_loaders.py` and `test_models.py` cases for checkpoint format and loader edge cases
- `README.md` expanded with architecture overview, data layout, and quick-start instructions.

### Fixed
- **Test isolation bug**: fixture teardown ordering caused cross-test state leakage; resolved in `conftest.py`.

### Changed
- Minor code fixes in `losses.py`, `stages.py`, `trainer.py`, `loaders.py`, and `evaluation/metrics.py` surfaced during test writing.
- `experiments/v1_*.yaml`: config keys cleaned up and documented inline.
- `requirements.txt` expanded with pinned versions.
- `preprocessing/case_matching/03_data_matching_v1.py` refactored.

---

## [2026-05-18] — Initial `src/` package; training pipeline

### Added
- **`src/` package** — monolithic notebook refactored into importable modules:
  - `src/models/reintervention.py` — `ReinterventionRiskModel`, `load_checkpoint`, `get_model`
  - `src/data/loaders.py` — MONAI pipeline, `PersistentDataset`, `RuntimeDataset`, cached + runtime transforms
  - `src/training/losses.py` — `ReinterventionLoss`, `MultiTaskReinterventionLoss`
  - `src/training/stages.py` — stage setup, optimizer construction, modality dropout
  - `src/training/trainer.py` — `run_pipeline`, `validate`, `log_xai_epoch`
  - `src/evaluation/metrics.py` — `compute_overall_metrics`, `save_metrics`
  - `src/evaluation/runner.py` — `run_test` (per-patient inference output)
  - `src/xai/explainer.py` — `ReinterventionXAI` (gradients, ablation, GradCAM)
- **Entry-point scripts**: `scripts/train.py`, `scripts/test.py`, `scripts/submit_slurm.sh` (SLURM wrapper with configurable GPU/memory/time)
- **Experiment configs**: `experiments/v1_no_complications.yaml`, `experiments/v1_with_complications.yaml`
- **Initial pytest suite** (`tests/`): `conftest.py` with shared fixtures and Merlin/PEFT mocks; `test_imports.py`, `test_loaders.py`, `test_losses.py`, `test_metrics.py`, `test_models.py`, `test_stages.py`, `test_xai.py`
- `.pytest.ini`, `.gitignore` updated

### Removed
- `notebooks/reintervention.py` (1 681-line monolith) — superseded by `src/`
- Stub files: `src/models/geometry.py`, `src/models/heads.py`, `src/data/preprocess.py`

---

## [2026-05-13] — First pre-operative model

### Added
- `ReinterventionRiskModel` — multi-modal architecture:
  - Visual branch: Merlin I3ResNet (frozen) + linear projection to 256-dim
  - Clinical branch: feature gate + MLP → 32-dim
  - AAA metrics branch: MLP → 32-dim
  - FiLM fusion (γ/β modulation, tanh-clamped ±0.1) + LayerNorm
  - Heteroscedastic uncertainty heads (`reintervention_head`, optional `complications_head`)
- `ReinterventionLoss` — focal BCE (γ=2, pos_weight=4.58) + heteroscedastic log-variance regularisation + L1 gate regularisation
- `MultiTaskReinterventionLoss` — adds masked per-complication focal losses (GT=-1 ignored)
- 2-stage training curriculum:
  - Stage A: Merlin frozen; clinical/AAA/fusion branches warm up on tabular data
  - Stage B: BN layers + visual projection unfrozen; scanner-domain adaptation
- Modality dropout (`p_clinical`, `p_aaa`) for causal training
- XAI: gradient-based clinical/AAA importance, modality ablation, GradCAM skeleton
- Best validation AUC: **0.80** (Stage B, no-complications variant)

---

## [2026-05-03] — Data preparation pipeline (v1)

### Added
- `preprocessing/case_matching/` — 6-step pre-operative data pipeline:
  - `00_dicom_to_nifti_renaming.py` — DICOM → NIfTI conversion and standardised naming
  - `01_cfd_name_cleaning.py` — CFD mesh filename normalisation
  - `02_umcg_name_cleaning.py` — UMCG-specific filename cleaning
  - `03_data_matching_v1.py` — cross-site patient matching
  - `04_data_organizing.py` — output layout organisation
  - `05_nnUNet_labels_metrics_generation.py` — nnUNet segmentation + AAA geometry metrics
  - `06_generate_data_split.py` — stratified train/val/test split JSON
- `preprocessing/generate_reports/` — LLM-based radiology report generation scripts
- `preprocessing/prepare_data.sh` — shell wrapper for the full pipeline

---

## [2026-04-09] — Project initialisation

### Added
- Repository skeleton: `src/`, `scripts/`, `experiments/`, `preprocessing/`, `tests/`, `documentation/` directories
- `README.md`, `LICENSE` (Apache 2.0), `.gitignore`
- Coarse `src/` stubs (`models/`, `data/`) as placeholders
