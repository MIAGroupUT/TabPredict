#!/bin/bash
#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=180:00:00
#SBATCH --partition=mia,mia-pof,am,main
#SBATCH --output=/home/ranema/AI4EVAR/logs/%x_%j.log
#SBATCH --error=/home/ranema/AI4EVAR/logs/%x_%j.err
#SBATCH --job-name=prepare_data

# -------------------------
# Load environment
# -------------------------
source /etc/profile.d/modules.sh
module load anaconda3/2025.06
export HTTPS_PROXY="http://proxy.utwente.nl:3128"

cd $HOME/AI4EVAR/ai4evar/data/case_matching

# -------------------------
# SLURM runtime env
# -------------------------
start_time=$SECONDS
echo $'\n--------------SLURM JOB STARTED-----------------'
export CUDNN_WORKSPACE_LIMIT=4096
export PYTORCH_ALLOC_CONF=expandable_segments:True

# -------------------------
# Activate environment
# -------------------------
source activate ai4evar

# -------------------------
# Run data preparation pipeline
# -------------------------
srun -u $HOME/.conda/envs/SIRE-opt/bin/python -m 00_dicom_to_nifti_renaming 
srun -u $HOME/.conda/envs/SIRE-opt/bin/python -m 01_cfd_name_cleaning 
srun -u $HOME/.conda/envs/SIRE-opt/bin/python -m 02_data_matching_v1 
srun -u $HOME/.conda/envs/SIRE-opt/bin/python -m 03_data_organizing 
srun -u $HOME/.conda/envs/SIRE-opt/bin/python -m 04_nnUNet_labels_metrics_generation \
    --input "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1/imagesTr" \
    --output-dir "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1" \
    --checkpoint-dir "/deepstore/datasets/mia/AI4EVAR/nnUNet/nnUNet_results/Dataset001_AAA_RADAR" \
    --device "cuda"
srun -u $HOME/.conda/envs/SIRE-opt/bin/python -m 05_generate_data_split 

echo $'\n--------------SLURM JOB FINISHED-----------------'
elapsed_time=$(($SECONDS - start_time))
echo "Elapsed time: $elapsed_time seconds"