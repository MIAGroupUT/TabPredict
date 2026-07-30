# Multi-Modal Risk Prediction for EndoVascular Aneurysm Repair

Submission to the [MICCAI 2026 MultiTab Workshop](https://multitab-miccai-2026.github.io).

**Paper:** _not yet published — link coming soon_

Multi-modal deep learning for post-operative re-intervention risk prediction in EVAR patients, combining CT volumes, clinical tabular data, and AAA geometry metrics from the **RADAR database** (542 AAA patients with imaging, 4 sites).

**Institution:** University of Twente

[![CI](https://github.com/MIAGroupUT/TabPredict/actions/workflows/ci.yml/badge.svg)](https://github.com/MIAGroupUT/TabPredict/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MIAGroupUT/TabPredict/gh-badges/tests.json)](https://github.com/MIAGroupUT/TabPredict/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MIAGroupUT/TabPredict/gh-badges/coverage.json)](https://github.com/MIAGroupUT/TabPredict/actions/workflows/ci.yml)
[![Last Commit](https://img.shields.io/github/last-commit/MIAGroupUT/TabPredict)](https://github.com/MIAGroupUT/TabPredict/commits/main)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

---

## Installation

```bash
conda activate tabpredict          # Python 3.11
export HTTPS_PROXY="http://proxy.utwente.nl:3128"   # required on HPC
pip install -r requirements.txt
```

---

## Quick Start

```bash
cd /home/ranema/TabPredict

# Train — MLP clinical backbone
python scripts/train.py --config experiments/mlp.yaml

# Train — SAINT clinical backbone
python scripts/train.py --config experiments/saint.yaml

# Train — TabM clinical backbone
python scripts/train.py --config experiments/tabm.yaml

# Train — classical ML backbones (LR | RF | GBM | XGBoost)
python scripts/train.py --config experiments/v1_logistic_regression.yaml
python scripts/train.py --config experiments/v1_random_forest.yaml
python scripts/train.py --config experiments/v1_gradient_boosting.yaml
python scripts/train.py --config experiments/v1_xgboost.yaml

# Evaluate on test split
python scripts/test.py \
    --config     experiments/mlp.yaml \
    --checkpoint /deepstore/datasets/mia/TabPredict/models/no_complications_MLP/best_stage_b.pt \
    --output_dir /deepstore/datasets/mia/TabPredict/test_results/no_complications_MLP

# Evaluate on all splits × all stage checkpoints
python scripts/test.py \
    --config     experiments/mlp.yaml \
    --output_dir /deepstore/datasets/mia/TabPredict/test_results/no_complications_MLP \
    --split all

# Run tests (421 tests, CPU-only, ~20 s)
~/.conda/envs/tabpredict/bin/python -m pytest tests/ -v
```

Override any config value at runtime without editing the YAML:

```bash
python scripts/train.py \
    --config experiments/mlp.yaml \
    --set training.batch_size=4 training.stages[0].epochs=2 wandb.mode=disabled
```

---

## Repository Structure

```
TabPredict/
├── experiments/                          # YAML experiment configs (one per backbone)
│   ├── mlp.yaml      # MLP backbone
│   ├── saint.yaml    # SAINT backbone
│   ├── tabm.yaml     # TabM backbone
│   ├── tabpfn.yaml   # TabPFN backbone
│   ├── v1_logistic_regression.yaml       # Logistic Regression backbone
│   ├── v1_random_forest.yaml             # Random Forest backbone
│   ├── v1_gradient_boosting.yaml         # Gradient Boosting backbone
│   └── v1_xgboost.yaml                   # XGBoost backbone
├── scripts/                              # Entry points
│   ├── train.py                          # Training
│   ├── test.py                           # Evaluation / inference
│   └── submit_slurm.sh                   # SLURM wrapper
├── src/                                  # Core library
│   ├── models/                           # ReinterventionRiskModel + tabular backbones
│   ├── data/                             # MONAI pipeline + loaders
│   ├── training/                         # Losses, stages, trainer
│   ├── xai/                              # Gradient-based explainability
│   └── evaluation/                       # Metrics + per-patient output
├── preprocessing/                        # Data preparation pipeline (00–07)
├── tests/                                # CPU-only pytest suite (421 tests)
└── documentation/                        # Technical documentation
```

---

## Documentation

| Document | Contents |
|---|---|
| [Architecture](documentation/architecture.md) | Model branches, FiLM fusion, output heads, forward API |
| [Models](documentation/models.md) | Tabular backbone comparison, projection inputs, imputation, per-backbone pipeline |
| [Training](documentation/training.md) | 2-stage curriculum, loss functions, optimizer, modality dropout |
| [Data](documentation/data.md) | Deepstore layout, clinical JSON, MONAI pipeline |
| [Preprocessing](documentation/preprocessing.md) | Raw DICOM → training-ready data, step-by-step pipeline |
| [Configuration](documentation/configuration.md) | Full YAML reference, all parameters explained |
| [XAI](documentation/xai.md) | Explainability methods, `full_explain()` output, GradCAM |
| [Evaluation](documentation/evaluation.md) | Inference, per-patient output files, all metrics |
| [Testing](documentation/testing.md) | Test suite coverage, gaps, how to run |
| [W&B Integration](documentation/wandb.md) | Logged metrics, tables, offline mode |
| [HPC / SLURM](documentation/hpc.md) | Job submission, cluster setup, monitoring |
