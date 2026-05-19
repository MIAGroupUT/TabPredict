# AI4EVAR — AI for EndoVascular Aneurysm Repair

Personalized risk prediction for EVAR patients using multi-modal deep learning. The model predicts post-operative **re-intervention risk** and individual **complication probabilities** from CT volumes, clinical tabular data, and AAA geometry metrics derived from the **RADAR database** (~1,000 AAA patients).

**Institution:** University of Twente &nbsp;|&nbsp; **W&B project:** `AAA-Reintervention-Risk`

---

## Installation

```bash
conda activate ai4evar          # Python 3.11
export HTTPS_PROXY="http://proxy.utwente.nl:3128"   # required on HPC
pip install -r requirements.txt
```

---

## Quick Start

```bash
cd /home/ranema/AI4EVAR/ai4evar

# Train — primary-task model (reintervention only)
python scripts/train.py --config experiments/v1_no_complications.yaml

# Train — multi-task model (reintervention + 7 complications)
python scripts/train.py --config experiments/v1_with_complications.yaml

# Evaluate a checkpoint
python scripts/test.py \
    --config     experiments/v1_no_complications.yaml \
    --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_no_complications/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications

# Run tests (CPU-only, ~20 s)
~/.conda/envs/ai4evar/bin/python -m pytest tests/ -v
```

Override any config value at runtime without editing the YAML:

```bash
python scripts/train.py \
    --config experiments/v1_no_complications.yaml \
    --set training.batch_size=4 training.stages[0].epochs=2 wandb.mode=disabled
```

---

## Repository Structure

```
ai4evar/
├── experiments/          # YAML experiment configs
│   ├── v1_no_complications.yaml
│   └── v1_with_complications.yaml
├── scripts/              # Entry points
│   ├── train.py          # Training
│   ├── test.py           # Evaluation / inference
│   └── submit_slurm.sh   # SLURM wrapper
├── src/                  # Core library
│   ├── models/           # ReinterventionRiskModel
│   ├── data/             # MONAI pipeline + loaders
│   ├── training/         # Losses, stages, trainer
│   ├── xai/              # Gradient-based explainability
│   └── evaluation/       # Metrics + per-patient output
├── preprocessing/        # Data preparation scripts (00–06)
├── tests/                # CPU-only pytest suite (220 tests)
└── documentation/        # Detailed technical documentation
```

---

## Documentation

| Document | Contents |
|---|---|
| [Architecture](documentation/architecture.md) | Model branches, FiLM fusion, output heads, forward API |
| [Training](documentation/training.md) | 2-stage curriculum, loss functions, optimizer, modality dropout |
| [Data](documentation/data.md) | Deepstore layout, clinical JSON, MONAI pipeline, preprocessing pipeline |
| [Configuration](documentation/configuration.md) | Full YAML reference, all parameters explained |
| [XAI](documentation/xai.md) | Explainability methods, `full_explain()` output, GradCAM |
| [Evaluation](documentation/evaluation.md) | Inference, per-patient output files, all metrics |
| [Testing](documentation/testing.md) | Test suite coverage, gaps, how to run |
| [W&B Integration](documentation/wandb.md) | Logged metrics, tables, offline mode |
| [HPC / SLURM](documentation/hpc.md) | Job submission, cluster setup, monitoring |

---

## Work Packages

**WP1 — Post-operative AI models** (this repository)
Re-intervention risk prediction using CTA volumes, clinical tabular data, and AAA geometry metrics. Two variants: primary-task-only and multi-task (complication-aware).

**WP2 — Shared Decision Making integration**
Integration of WP1 model outputs and XAI explanations into clinical SDM tools and patient-facing dashboards.

**WP3 — Health Technology Assessment**
Early HTA of the AI-driven personalised follow-up protocol using test-set metrics and confidence levels.
