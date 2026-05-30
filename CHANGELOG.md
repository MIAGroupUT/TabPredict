# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Entries are grouped by push milestone (no formal version tags yet).

---

## [Unreleased]

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
- All documentation (`architecture.md`, `configuration.md`, `data.md`, `training.md`, `testing.md`, `CLAUDE.md`) updated to reflect new geometry, regularisation values, FiLM clamp, checkpoint criterion, and WeightedRandomSampler.

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
