# HPC / SLURM

## Cluster Setup

| Setting | Value |
|---|---|
| Scheduler | SLURM |
| Partition | `main`, `mia`, `mia-pof`, `am` |
| Resources | 32 CPUs, 1 GPU, 180 GB RAM, 180h wall time |
| Conda env | `/home/ranema/.conda/envs/ai4evar` (Python 3.11) |
| Proxy | `HTTPS_PROXY=http://proxy.utwente.nl:3128` (required for W&B and package downloads) |
| Logs | `/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/<job_name>_<job_id>.log` |
| Checkpoints | `/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/models/<experiment_name>/` |

The proxy is set automatically by `scripts/submit_slurm.sh`.

---

## SLURM Wrapper — `scripts/submit_slurm.sh`

All jobs are submitted through the wrapper. It:
1. Sets the conda environment and proxy
2. Changes to the `ai4evar_multitab/` directory
3. Calls either `scripts/train.py` or `scripts/test.py` with the provided arguments

Usage:

```bash
sbatch [slurm-options] scripts/submit_slurm.sh train|test [script-args...]
```

---

## Training Jobs

```bash
sbatch \
    --job-name=mt_mlp \
    --output=/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/mt_mlp_%j.log \
    scripts/submit_slurm.sh train \
    --config experiments/v1_no_complications_mlp.yaml
```

Override config values at submission time:

```bash
sbatch scripts/submit_slurm.sh train \
    --config experiments/v1_no_complications_mlp.yaml \
    --set training.batch_size=4 wandb.mode=offline
```

---

## Test / Inference Jobs

```bash
sbatch \
    --job-name=test_mlp \
    --output=/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/test_mlp_%j.log \
    scripts/submit_slurm.sh test \
    --config     experiments/v1_no_complications_mlp.yaml \
    --checkpoint /deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/models/no_complications_MLP/best_stage_b.pt \
    --output_dir /deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/test_results/no_complications_MLP \
    --gradcam
```

---

## Data Preparation Jobs

Scripts live in `preprocessing/case_matching_v1/` and must be run in order (00 → 07).
See **[Preprocessing](preprocessing.md)** for the full technical description of each step.

```bash
cd /home/ranema/AI4EVAR/ai4evar_multitab/preprocessing/case_matching_v1

# Step 00 — DICOM → NIfTI (all sessions, all sites)
python 00_dicom_to_nifti_renaming.py

# Step 01 — Build session index JSON
python 01_build_session_index.py

# Step 02 — Rename CFD meshes to match NIfTI naming
python 02_cfd_name_cleaning.py

# Step 03 — Match imaging sessions to SPSS clinical records
python 03_data_matching_v1.py

# Step 04 — Organise NIfTIs into training_subset_v1/
python 04_data_organizing.py

# Step 05 — nnUNet segmentation + distance maps + AAA metrics
python 05_nnUNet_labels_metrics_generation.py

# Step 06 — Crop full-body scans to aortic region
python 06_crop_full_body_scans.py

# Step 07 — Stratified train/val/test split
python 07_generate_data_split.py
```

---

## Monitoring

```bash
# View running jobs
squeue -u ranema

# Live log tail
tail -f /deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/*.log

# Watch a specific job
watch -n 10 squeue -u ranema -j <job_id>

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u ranema
```

---

## GPU Availability

Check available GPUs before submitting:

```bash
sinfo -p mia --format="%N %G %t"
```

---

## Common Issues

**W&B not syncing:** Check that the proxy is set. On the login node: `export HTTPS_PROXY=http://proxy.utwente.nl:3128` then test with `curl -I https://api.wandb.ai`.

**Out of memory:** The most common cause is RAM exhaustion from persistent DataLoader workers accumulating RSS over epochs (Python copy-on-write degradation). Default is `num_workers=4` with `prefetch_factor=1`. If kills still occur, reduce via `--set training.num_workers=2`. Also consider reducing `training.batch_size` or `training.train_cache_rate`.

**Job pending too long:** Try a different partition (`--partition mia-pof` or `--partition am`) or reduce resource requests.

**MONAI cache corruption:** If training crashes mid-epoch, the persistent cache may be partially written. Delete the cache directory for the affected stage and restart:
```bash
rm -rf /deepstore/datasets/mia/AI4EVAR/derivatives/monai_cache/train_B/
```
