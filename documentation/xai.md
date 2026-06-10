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

The top-20 features by absolute gradient magnitude are returned. Scores are normalised to [0, 1] within the clinical group (divided by the max gradient across all clinical features). Categorical levels follow the same thresholds as AAA importance: `"HIGH"` (score > 0.7), `"MED"` (score > 0.3), `"LOW"` (otherwise).

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

## Technical Implementation

### Confidence — formula and calibration

```python
reint_conf = 2.0 * abs(reint_prob - 0.5)
```

Confidence is the **absolute distance from the decision boundary** (0.5), scaled so it spans [0, 1]. It is **self-calibrating** — no post-hoc temperature scaling or Platt scaling is applied (temperature defaults to 1.0 unless passed explicitly). The model's probability is therefore used as-is from `sigmoid(logit)`.

| prob | confidence | level |
|------|-----------|-------|
| 0.95 | 0.90 | HIGH |
| 0.75 | 0.50 | MED |
| 0.55 | 0.10 | LOW |
| 0.50 | 0.00 | LOW |

Level thresholds: HIGH > 0.6, MED > 0.3, LOW ≤ 0.3.

---

### GradCAM — step-by-step

Target layer: **`I3ResNet.layer4[-1]`** — the last residual block of the last ResNet stage inside the Merlin backbone. This is found dynamically by `_get_i3res()`, which walks `model.visual_branch.named_modules()` looking for a module that has both `layer1` and `layer4` attributes (handles LoRA/PEFT wrapping transparently).

**Gradient checkpointing patch**

Gradient checkpointing is active during training to reduce memory. It breaks gradient flow by re-running the forward pass segment during backward instead of storing activations. GradCAM needs activations **and** their gradients at the same time, so before the forward pass the checkpointing function is monkey-patched to a no-op:

```python
cp_mod.checkpoint = lambda fn, *args, **kwargs: fn(*args)
```

This is restored in the `finally` block regardless of errors.

**Forward hook**

A hook is registered on the target layer before the forward pass:

```python
def fwd_hook(module, inp, out):
    store["feat"] = out        # activations: shape [B, C, D, H, W]
    if out.requires_grad:
        out.retain_grad()      # prevent intermediate grad from being freed
```

**Backward**

```python
out[0].sum().backward()  # grad w.r.t. sum of reintervention logits
```

This computes `∂ logit / ∂ feat` via automatic differentiation. Using `.sum()` for a batch of 1 is equivalent to `.squeeze().backward()`.

**Weight computation and CAM**

```python
weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # global spatial average per channel
cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
```

- `weights` shape: `[B, C, 1, 1, 1]` — one scalar weight per channel, computed as the mean gradient across all spatial positions. This is the global average pooling step from the original GradCAM paper.
- `(weights * acts).sum(dim=1)` — weighted sum across channels, producing `[B, 1, D', H', W']` where D'×H'×W' is the spatial resolution of layer4 (much smaller than the input).
- `ReLU` zeroes negative values — only regions that **increase** the logit are retained.

**Upsampling and normalisation**

```python
cam_up = F.interpolate(cam, size=cta.shape[2:], mode="trilinear", align_corners=False)
cam_np = cam_up.squeeze().numpy()
cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
```

Trilinear upsampling back to the full CTA volume size (224×224×160). Min-max normalisation maps to [0, 1]. The `1e-8` epsilon prevents division-by-zero for flat CAMs (all zeros → returns zeros).

**Rendering**

Slices are taken at the volume centre along each axis (D//2, H//2, W//2). The data is in RAS orientation (`Orientationd(axcodes='RAS')`), so axes are: 0=R, 1=A, 2=S. Each 2D slice is anatomically oriented via `flipud(T)` before display.

Overlay layers (in render order):
1. CT: gray colormap, `vmin=0 vmax=1` (HU-clipped and normalised during preprocessing)
2. GradCAM: jet colormap, `alpha=0.4`
3. Lumen mask: RGBA cyan `[0, 1, 1, 0.35]`
4. ILT mask: RGBA yellow `[1, 1, 0, 0.35]`

---

### Clinical & AAA gradient importance — limitations

Both methods use **input-gradient saliency**: the magnitude of `∂ sigmoid(logit) / ∂ input_i`. Key properties:

- **Local**: computed at a single input point. A nearby input with the same clinical values could have a different gradient if the model's decision surface is nonlinear.
- **First-order only**: does not capture non-linear interactions between features. If two features interact multiplicatively, neither individually may show high gradient.
- **Magnitude only, not direction**: a HIGH-impact feature could be pushing probability **up or down**. The sign of the gradient is not reported. To get direction you would need `clin_var.grad[i]` (signed), not `abs()`.
- **Not normalised across patients**: scores are normalised within a single call (divided by that call's max gradient). A score of 0.8 for patient A is not comparable to 0.8 for patient B.

---

### Modality importance — ablation vs gradient

Unlike clinical/AAA importance, modality importance uses **zero-ablation**, not gradients. Zero-ablation replaces an entire modality's input tensor with zeros and measures the absolute probability shift.

This is more informative for **modality-level** attribution because:
- Gradients w.r.t. a high-dimensional input (CTA: 4×224×224×160 ≈ 36M values) would need to be aggregated somehow — any aggregation introduces choices.
- Zero-ablation directly answers "what does the model do without this input?" in a model-agnostic way.

**Caveat**: replacing CTA with zeros ≠ "CTA is missing". Zero is a specific value (HU=0 after clipping/normalising, roughly water density). The model has not been trained with masked CTA inputs, so its output under this perturbation reflects learned behaviour at an out-of-distribution input. The ablation impact is a proxy for modality contribution, not a causal estimate.

---

### Causal chain — template logic, not learned

The causal chain string is **not generated by a language model**. It is a deterministic template filled with values from the other XAI outputs:

```
Slots:
  top_clin  = clin_imp[0][0]          (highest-gradient clinical feature name)
  top_mod   = argmax(modality_imp)     (dominant modality by ablation fraction)
  reint_prob                           (from sigmoid output)
  top_comps = sorted comp_probs descending, filtered to > 0.15, top 3
              (multi-task model only)
```

Template selection (no-complications model):
```
if reint_prob > 0.5:
    "Elevated <top_clin> detected by <top_mod> → likely reintervention needed (<reint_prob>%)"
else:
    "<top_clin> via <top_mod> — monitoring advised, low reintervention risk (<reint_prob>%)"
```

Template selection (complications model, reint_prob > 0.5):
```
"Elevated <top_clin> via <top_mod> → probable complications: <comp1> (X%, CONF), ... → likely reintervention (Y%)"
```

**Implication**: the causal chain is only as good as the gradient-based top clinical feature and the ablation-based top modality. If those are noisy or unintuitive, the narrative will be too.

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
