# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Entries are grouped by push milestone (no formal version tags yet).

---

## [Unreleased]

_No unreleased changes._

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
