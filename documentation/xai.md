# XAI — Explainability

## Overview

`ReinterventionXAI` provides post-hoc, gradient-based explanations at inference time. No retraining or architectural changes are needed. The main entry point is `full_explain()`, which returns a single dict covering all explanation types.

```python
from src.xai.explainer import ReinterventionXAI

xai = ReinterventionXAI(model, feature_names, complications=[...])
result = xai.full_explain(cta, clin, aaa)
```

> Before running XAI, call `disable_checkpointing(model)` to ensure gradients can flow through the entire backbone.

---

## `full_explain()` Output

| Key | Type | Description |
|---|---|---|
| `reintervention_prob` | float | Reintervention probability, `sigmoid(logit)` |
| `reintervention_confidence` | float | `2 × \|prob − 0.5\|` — 0 when uncertain (prob≈0.5), 1 when certain (prob near 0 or 1) |
| `confidence_level` | str | `"HIGH"` (>0.60), `"MED"` (>0.30), `"LOW"` (≤0.30) |
| `clinical_importance` | list of (name, score, level) | Top-20 clinical features by gradient magnitude |
| `aaa_importance` | list of (name, score, level) | All 5 AAA geometry metrics, normalised |
| `modality_importance` | dict | CTA / Clinical / AAA_metrics ablation fractions (sum to 1.0) |
| `causal_chain` | str | Human-readable narrative: top driver → predicted outcome |
| `primary_risk_factor` | str | Single most important feature or complication |
| `complication_probs` | dict | Per-complication probabilities (multi-task only) |
| `complication_confidence` | dict | Per-complication confidence levels (multi-task only) |

---

## Explanation Methods

### Clinical Importance

Computes the gradient of the predicted reintervention probability with respect to each clinical input feature:

```
importance_i = |∂ sigmoid(logit) / ∂ clin_i|
```

The top-20 features by absolute gradient magnitude are returned, together with their importance score (normalised to [0, 1] within the batch) and a categorical level (`"HIGH"` / `"MED"` / `"LOW"`).

This tells you which clinical features most influenced the specific prediction for this patient — not a global feature ranking.

### AAA Importance

Same gradient approach applied to the 5 AAA geometry scalar inputs. All 5 are returned, sorted by normalised importance.

### Modality Importance — Ablation

Each input modality is zeroed out independently (replaced by all-zeros) and the change in predicted probability is measured:

```
Δ_cta      = |prob_original − prob_without_cta|
Δ_clinical = |prob_original − prob_without_clinical|
Δ_aaa      = |prob_original − prob_without_aaa|
```

These deltas are normalised so they sum to 1.0:

```python
result["modality_importance"] = {
    "CTA":         Δ_cta      / (Δ_cta + Δ_clinical + Δ_aaa),
    "Clinical":    Δ_clinical / ...,
    "AAA_metrics": Δ_aaa      / ...,
}
```

A high CTA fraction means the CTA scan was the dominant driver of this patient's prediction. A high Clinical fraction means the tabular data dominated.

### Causal Chain (Narrative)

`full_explain()` generates a short, human-readable string summarising the key driver:

```
"Elevated Bl_AAAdiam (HIGH impact) suggests high re-intervention risk (72%)"
```

In the multi-task variant, dominant complications are also woven into the narrative:

```
"endoleak_2 (HIGH, 65%) and elevated Bl_AAAdiam → likely re-intervention (72%)"
```

---

## GradCAM — Spatial Heatmap

`xai.gradcam(cta, clin, aaa)` hooks into the last ResNet block of the Merlin **I3ResNet** backbone to produce a 3D spatial heatmap:

```python
cam = xai.gradcam(cta, clin, aaa)
# Returns: np.ndarray of shape (D, H, W), values in [0, 1]
# Returns None if the backbone is not an I3ResNet (e.g., the mock used in tests)
```

### Saving GradCAM PNGs

```python
xai.save_gradcam_pngs(cta, cam, out_dir)
```

Writes four PNG files to `out_dir`:

| File | Content |
|---|---|
| `gradcam_combined.png` | 3-panel figure (axial / sagittal / coronal) at 150 DPI |
| `gradcam_axial.png` | Central axial slice with overlay |
| `gradcam_sagittal.png` | Central sagittal slice |
| `gradcam_coronal.png` | Central coronal slice |

Each slice shows: CT greyscale → GradCAM heatmap (jet colormap, α=0.4) → lumen mask (cyan) → ILT mask (yellow).

---

## XAI During Training

Every `xai_every_n_epochs` epochs, `log_xai_epoch()` runs `full_explain()` on `xai_samples` validation patients and logs the results to W&B.

Logged W&B table columns:

```
epoch | stage | case_idx | reint_prob | confidence_level |
primary_risk_factor | causal_chain |
imp_CTA | imp_Clinical | imp_AAA |
[comp_<name> for each complication]
```

If `xai_gradcam: true`, GradCAM slices (axial / sagittal / coronal) are additionally logged as W&B images alongside the table.

See [W&B Integration](wandb.md) for table names and logging frequency.

---

## Usage at Inference

`scripts/test.py` runs `full_explain()` on every patient in the evaluation split and saves results to the output directory:

```
<output_dir>/<patient_id>/
├── xai.json          # full full_explain() output dict
├── xai_report.txt    # human-readable clinical report
└── gradcam_*.png     # (only with --gradcam flag)
```

See [Evaluation](evaluation.md) for the full output structure.
