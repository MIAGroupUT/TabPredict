# HPC / SLURM

## Cluster Setup

| Setting | Value |
|---|---|
| Scheduler | SLURM |
| Partition | `main`, `mia`, `mia-pof`, `am` |
| Resources | 32 CPUs, 1 GPU, 180 GB RAM, 180h wall time |
| Conda env | `/home/ranema/.conda/envs/ai4evar` (Python 3.11) |
| Proxy | `HTTPS_PROXY=http://proxy.utwente.nl:3128` (required for W&B and package downloads) |
| Logs | `/home/ranema/AI4EVAR/logs/<job_name>_<job_id>.log` |
| Checkpoints | `/home/ranema/AI4EVAR/checkpoints/<experiment_name>/` |

The proxy is set automatically by `scripts/submit_slurm.sh`.

---

## SLURM Wrapper — `scripts/submit_slurm.sh`

All jobs are submitted through the wrapper. It:
1. Sets the conda environment and proxy
2. Changes to the `ai4evar/` directory
3. Calls either `scripts/train.py` or `scripts/test.py` with the provided arguments

Usage:

```bash
sbatch [slurm-options] scripts/submit_slurm.sh train|test [script-args...]
```

---

## Training Jobs

```bash
# Primary-task model (reintervention only)
sbatch \
    --job-name=v1_no_comp \
    --output=/home/ranema/AI4EVAR/logs/v1_no_comp_%j.log \
    scripts/submit_slurm.sh train \
    --config experiments/v1_no_complications.yaml

# Multi-task model (reintervention + 7 complications)
sbatch \
    --job-name=v1_comp \
    --output=/home/ranema/AI4EVAR/logs/v1_comp_%j.log \
    scripts/submit_slurm.sh train \
    --config experiments/v1_with_complications.yaml
```

Override config values at submission time:

```bash
sbatch scripts/submit_slurm.sh train \
    --config experiments/v1_no_complications.yaml \
    --set training.batch_size=4 wandb.mode=offline
```

---

## Test / Inference Jobs

```bash
# Primary-task model
sbatch \
    --job-name=test_no_comp \
    --output=/home/ranema/AI4EVAR/logs/test_no_comp_%j.log \
    scripts/submit_slurm.sh test \
    --config     experiments/v1_no_complications.yaml \
    --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_no_complications/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications \
    --gradcam

# Multi-task model
sbatch \
    --job-name=test_comp \
    --output=/home/ranema/AI4EVAR/logs/test_comp_%j.log \
    scripts/submit_slurm.sh test \
    --config     experiments/v1_with_complications.yaml \
    --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_with_complications/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_with_complications \
    --gradcam
```

---

## Data Preparation Jobs

Scripts live in `preprocessing/case_matching_v1/` and must be run in order (00 → 07).
See **[Preprocessing](preprocessing.md)** for the full technical description of each step.

```bash
cd /home/ranema/AI4EVAR/ai4evar/preprocessing/case_matching_v1

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
tail -f /home/ranema/AI4EVAR/logs/*.log

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

**Out of memory:** The most common cause is RAM exhaustion from persistent DataLoader workers accumulating RSS over epochs (Python copy-on-write degradation). Current defaults are `num_workers=4` (`v1_no_complications`) and `num_workers=6` (`v1_with_complications`) with `prefetch_factor=1`. If kills still occur, reduce further via `--set training.num_workers=2`. Also consider reducing `training.batch_size` or `training.train_cache_rate`.

**Job pending too long:** Try a different partition (`--partition mia-pof` or `--partition am`) or reduce resource requests.

**MONAI cache corruption:** If training crashes mid-epoch, the persistent cache may be partially written. Delete the cache directory for the affected stage and restart:
```bash
rm -rf /deepstore/datasets/mia/AI4EVAR/derivatives/monai_cache/train_B/
```
