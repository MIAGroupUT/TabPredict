# Evaluation

## Running Inference

`scripts/test.py` evaluates a trained checkpoint on a data split and writes structured per-patient output.

```bash
python scripts/test.py \
    --config     experiments/v1_no_complications.yaml \
    --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_no_complications/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications \
    [--split     test]      # "train" | "val" | "test" (default: "test")
    [--gradcam]             # also compute and save GradCAM PNGs (slower)
    [--workers   8]         # DataLoader workers
```

The checkpoint contains all necessary normalisation stats (`clin_mean`, `clin_std`, `feature_names`) and the optimal decision threshold (`threshold`), so no separate config is needed for those.

### On SLURM

```bash
sbatch --job-name=test_no_comp scripts/submit_slurm.sh test \
    --config     experiments/v1_no_complications.yaml \
    --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_no_complications/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications \
    --gradcam
```

---

## Output Directory Structure

```
<output_dir>/
├── test_metrics.json         # overall metrics (AUC, F1, confusion matrix, ...)
├── test_metrics.csv          # same, flattened (comp_auc_detail expanded to rows)
├── test_log.txt              # timestamped console log
└── <patient_id>/             # one folder per patient
    ├── prediction.json       # probability, prediction, ground truth, threshold
    ├── xai.json              # full full_explain() output
    ├── xai_report.txt        # human-readable clinical narrative
    └── gradcam_*.png         # GradCAM slices (only with --gradcam)
```

---

## Per-Patient Files

### `prediction.json`

```json
{
  "patient_id":          "0042_RNST",
  "reintervention_prob": 0.68,
  "reintervention_pred": 1,
  "reintervention_gt":   1,
  "label_type":          "TP",
  "threshold":           0.52,
  "logvar":             -1.3
}
```

- `label_type` — one of `"TP"`, `"FP"`, `"TN"`, `"FN"`
- `threshold` — the F-beta optimal threshold loaded from the checkpoint
- `logvar` — raw log-variance output (lower = more confident)

### `xai.json`

Full `full_explain()` output dict. See [XAI](xai.md) for field descriptions.

### `xai_report.txt`

Human-readable plain-text clinical report. Example structure:

```
Patient: 0042_RNST
Reintervention probability: 68%  [HIGH confidence]
Prediction: POSITIVE  |  Ground truth: POSITIVE  |  Label: TP

PRIMARY RISK FACTOR: Bl_AAAdiam (HIGH impact)

TOP CLINICAL DRIVERS:
  1. Bl_AAAdiam          0.89  HIGH
  2. Bl_NeckLen          0.43  MED
  ...

AAA GEOMETRY:
  volume_lumen_ml        0.71  HIGH
  max_lumen_to_wall_dist 0.58  MED
  ...

MODALITY IMPORTANCE:
  CTA:          62%
  Clinical:     31%
  AAA metrics:   7%

CAUSAL SUMMARY:
  Elevated Bl_AAAdiam (HIGH impact) → likely re-intervention (68%)
```

---

## Overall Metrics — `test_metrics.json`

| Metric | Description |
|---|---|
| `n_samples` | Total number of patients evaluated |
| `n_positive` / `n_negative` | Class counts |
| `auc` | AUROC |
| `f1` | F1 score at the threshold |
| `precision` | Positive predictive value |
| `recall` | Sensitivity |
| `accuracy` | Overall accuracy |
| `threshold_used` | Decision threshold (from checkpoint) |
| `best_threshold` | Optimal F-beta threshold on this split's PR curve |
| `best_fbeta` | F-beta score at best threshold |
| `TPR_sensitivity` | True positive rate |
| `TNR_specificity` | True negative rate |
| `TP`, `FP`, `TN`, `FN` | Confusion matrix entries |
| `comp_auc_detail` | Dict of per-complication AUROC (multi-task only) |
| `comp_mean_auc` | Mean of per-complication AUROCs (multi-task only) |

### `test_metrics.csv`

Same data, but `comp_auc_detail` is expanded: each `{comp_name: value}` entry becomes a row with key `auc_<comp_name>`. This makes the CSV flat and easy to import into a spreadsheet.

---

## Interpreting Results

### Threshold

The checkpoint stores the F-beta (β=2) optimal threshold computed on the validation set. This threshold weights **recall 4× over precision** — the clinical requirement that missing a true re-intervention is worse than a false alarm.

When `threshold_used` and `best_threshold` differ significantly on the test set, consider whether the validation distribution matches the test distribution.

### Confidence Levels

| Level | Condition |
|---|---|
| `HIGH` | `1 / (1 + σ) ≥ 0.75` |
| `MED` | `1 / (1 + σ) ≥ 0.50` |
| `LOW` | `1 / (1 + σ) < 0.50` |

Where σ = `exp(0.5 × logvar)`.

Low-confidence predictions should be flagged for clinical review regardless of the binary prediction.

### Complication AUC

Per-complication AUROCs are only computed for complications with at least 2 unique ground-truth values in the evaluated split. If all patients in the split have the same ground-truth label for a complication (e.g. all endoleak_5=0), that complication is skipped and excluded from `comp_mean_auc`.
