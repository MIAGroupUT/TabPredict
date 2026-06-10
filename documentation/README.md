# Documentation

Technical reference for the AI4EVAR project. Each document covers one concern; start with **Architecture** if you're new to the codebase.

---

## Getting oriented

```
New to the project?   → Architecture → Training → Data
Setting up data?      → Preprocessing → Data
Picking a backbone?   → Models
Running on the HPC?   → HPC / SLURM → Configuration
Debugging a run?      → W&B Integration → Evaluation
Adding tests?         → Testing
Reading XAI output?   → XAI
```

---

## Index

### Model & Training

| Document | One-line summary |
|----------|-----------------|
| [Architecture](architecture.md) | `ReinterventionRiskModel` internals — visual branch, tabular branch, FiLM fusion, output heads |
| [Models](models.md) | All `clinical_branch` options compared: differentiability, feature gate, imputation, projection inputs, fit timing, param count |
| [Training](training.md) | 2-stage curriculum (Stage A / B), loss functions, optimizer, modality dropout, feature gate regularisation |
| [Configuration](configuration.md) | Full YAML reference — every key explained with defaults |

### Data & Preprocessing

| Document | One-line summary |
|----------|-----------------|
| [Preprocessing](preprocessing.md) | Raw DICOM → training-ready deepstore layout, step-by-step (scripts 00–06) |
| [Data](data.md) | Deepstore directory layout, clinical JSON schema, MONAI transform pipeline |

### Evaluation & Explainability

| Document | One-line summary |
|----------|-----------------|
| [Evaluation](evaluation.md) | Running inference, per-patient output files, all reported metrics |
| [XAI](xai.md) | `full_explain()` output, gradient-based clinical importance, GradCAM spatial heatmaps |

### Infrastructure

| Document | One-line summary |
|----------|-----------------|
| [HPC / SLURM](hpc.md) | Job submission, cluster environment, proxy setup, monitoring logs |
| [W&B Integration](wandb.md) | What is logged, prediction tables, offline mode, run naming |
| [Testing](testing.md) | Test suite layout, coverage by module, how to run locally |

---

## Key numbers (quick reference)

| | Value |
|-|-------|
| Patients | ~1,000 AAA (RADAR database) |
| Clinical features (after filter) | ~42 (from 240 raw, <50% missing) |
| AAA geometry scalars | 5 |
| CTA input shape | 4 × 224 × 224 × 160 (CT + lumen + ILT + dist map) |
| Fusion dim | 288 (256 visual + 32 tabular) |
| Test suite | 350+ tests, CPU-only, ~40 s |
| W&B project | `AAA-Reintervention-Risk` |
