# W&B Integration

## Project

All runs log to the W&B project **`AAA-Reintervention-Risk`**. The run name and mode are set in the YAML config:

```yaml
wandb:
  project: AAA-Reintervention-Risk
  run_name: v1_no_complications
  mode: online       # "online" | "offline" | "disabled"
```

Override at runtime:

```bash
python scripts/train.py --config ... --set wandb.mode=offline
```

---

## Modes

| Mode | Behaviour |
|---|---|
| `online` | Syncs to W&B cloud in real time. Requires network access (use proxy on HPC). |
| `offline` | Buffers locally in `wandb/` directory. Sync later with `wandb sync wandb/offline-run-*/`. |
| `disabled` | W&B is fully disabled; `wandb.log` is a no-op. No local files written. Use for quick local tests. |

---

## Metrics Logged Per Epoch

All metrics are logged at the end of each validation epoch.

### Reintervention Metrics (`val/` prefix)

| Key | Description |
|---|---|
| `val/loss_total` | Total validation loss |
| `val/reint_auc` | AUROC for reintervention prediction |
| `val/reint_f1` | F1 at current threshold |
| `val/reint_precision` | Precision at current threshold |
| `val/reint_recall` | Recall (sensitivity) at current threshold |
| `val/reint_accuracy` | Accuracy at current threshold |
| `val/reint_best_thresh` | F2-optimal threshold on this epoch's val PR curve; also stored in checkpoint |
| `val/mean_confidence` | Mean `2×\|prob−0.5\|` across val patients (0 = uncertain, 1 = certain) |
| `val/reint_TP` / `val/reint_FP` / `val/reint_TN` / `val/reint_FN` | Confusion matrix |
| `val/reint_TPR` / `val/reint_TNR` / `val/reint_FPR` / `val/reint_FNR` | Derived rates |

### Complication Metrics (multi-task variant only)

| Key | Description |
|---|---|
| `val/auc_<comp_name>` | Per-complication AUROC (one per complication) |
| `val/comp_mean_auc` | Mean of per-complication AUROCs |

### Training Loss (`metrics/` prefix, logged per training step)

| Key | Description |
|---|---|
| `metrics/loss_total` | Batch total loss |
| `metrics/loss_reint` | Reintervention component |
| `metrics/loss_comp` | Complication component (multi-task only) |

### Learning Rates (`lr/` prefix)

One entry per optimizer parameter group (e.g. `lr/clinical`, `lr/i3resnet_bn`, `lr/feature_gate`, `lr/cta_adapter`). Useful for verifying the cosine schedule is working correctly.

### Stage Loss (`stage_A/` or `stage_B/` prefix)

| Key | Description |
|---|---|
| `stage_A/train_loss` | Training loss averaged over the epoch (Stage A) |
| `stage_A/val_loss` | Validation loss (Stage A) |
| `stage_B/train_loss` | Same for Stage B |
| `stage_B/val_loss` | |

---

## Tables

### Prediction Matrix — `eval/full_matrix_stage_<X>`

Logged every **3 epochs**. Contains one row per validation patient:

| Column | Description |
|---|---|
| `case_id` | Patient identifier |
| `epoch` | Current epoch |
| `stage` | A or B |
| `reint_prob` | Predicted reintervention probability |
| `reint_pred` | Binary prediction at threshold |
| `reint_gt` | Ground truth |
| `label_type` | TP / FP / TN / FN |
| `comp_prob_<name>` | Per-complication probability (multi-task only) |

### XAI Table — `xai/table_stage_<X>`

Logged every `xai_every_n_epochs` epochs (default: 3). Contains one row per explained patient:

| Column | Description |
|---|---|
| `epoch` | Current epoch |
| `stage` | A or B |
| `case_idx` | Patient index in val set |
| `reint_prob` | Predicted probability |
| `confidence_level` | HIGH / MED / LOW |
| `primary_risk_factor` | Most important feature or complication |
| `causal_chain` | Human-readable narrative |
| `imp_CTA` | CTA ablation importance fraction |
| `imp_Clinical` | Clinical ablation importance fraction |
| `imp_AAA` | AAA ablation importance fraction |
| `comp_<name>` | Per-complication probability (multi-task only) |

If `xai_gradcam: true`, GradCAM slices (axial, sagittal, coronal) are additionally logged as W&B images.

---

## Offline Sync

After a SLURM run with `wandb.mode=offline`, sync the buffered data:

```bash
cd /home/ranema/TabPredict
wandb sync wandb/offline-run-*/
```

The `wandb/` directory is gitignored and lives locally in the repo root.
