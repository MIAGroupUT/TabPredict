#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job submission — AI4EVAR training / testing
#
# Usage (from repo root ai4evar/):
#
#   Training:
#     sbatch --job-name=v1_no_comp  scripts/submit_slurm.sh train \
#         --config experiments/v1_no_complications.yaml
#
#     sbatch --job-name=v1_comp     scripts/submit_slurm.sh train \
#         --config experiments/v1_with_complications.yaml
#
#   Testing — all stage checkpoints automatically (stage_a + stage_b):
#     sbatch --job-name=test_no_comp scripts/submit_slurm.sh test \
#         --config     experiments/v1_no_complications.yaml \
#         --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications \
#         --gradcam
#
#   Testing — single checkpoint only:
#     sbatch --job-name=test_no_comp scripts/submit_slurm.sh test \
#         --config     experiments/v1_no_complications.yaml \
#         --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_no_complications/best_stage_b.pt \
#         --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications \
#         --gradcam
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ai4evar
#SBATCH -c 32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=180:00:00
#SBATCH --partition=main,mia,mia-pof,am
#SBATCH --output=/home/ranema/AI4EVAR/logs/%x_%j.log
#SBATCH --error=/home/ranema/AI4EVAR/logs/%x_%j.err

set -euo pipefail

# ── environment ───────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
module load anaconda3/2025.06

export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1

PYTHON="$HOME/.conda/envs/ai4evar/bin/python"

# Always run from the repo root, regardless of where sbatch is called from
cd "$HOME/AI4EVAR/ai4evar"
mkdir -p /home/ranema/AI4EVAR/logs

# ── dispatch ──────────────────────────────────────────────────────────────────
MODE="${1:-train}"
shift  # consume the mode; remaining $@ is forwarded to the Python script

start_time=$SECONDS
echo $'\n--------------SLURM JOB STARTED-----------------'
echo "Mode   : ${MODE}"
echo "Args   : $*"
echo "Host   : $(hostname)"
echo "GPUs   : ${CUDA_VISIBLE_DEVICES:-unset}"
echo ""

case "${MODE}" in
    train)
        srun -u "${PYTHON}" scripts/train.py "$@"
        ;;
    test)
        srun -u "${PYTHON}" scripts/test.py "$@"
        ;;
    *)
        echo "Unknown mode '${MODE}'. Use 'train' or 'test'."
        exit 1
        ;;
esac

echo $'\n--------------SLURM JOB FINISHED-----------------'
elapsed=$(( SECONDS - start_time ))
echo "Elapsed time: ${elapsed}s  ($(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s)"
