#!/bin/bash
#SBATCH -c 12
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=180G
#SBATCH --time=180:00:00
#SBATCH --partition=mia,mia-pof,am
#SBATCH --output=/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/preprocessing_v1_multitab/06_crop_full_body_scans_V1.log
#SBATCH --error=/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/preprocessing_v1_multitab/06_crop_full_body_scans_V1.err
#SBATCH --job-name=prepare_data

#--output=/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/preprocessing_v1_multitab/%x_%j.log
#--error=/deepstore/datasets/mia/AI4EVAR/ai4evar_multitab/logs/preprocessing_v1_multitab/%x_%j.err


# -------------------------
# Load environment
# -------------------------
source /etc/profile.d/modules.sh
module load anaconda3/2025.06
export HTTPS_PROXY="http://proxy.utwente.nl:3128"

cd $HOME/AI4EVAR/ai4evar_multitab/preprocessing/case_matching_v1
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
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m 00_dicom_to_nifti_renaming 
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m 01_build_session_index
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m 02_cfd_name_cleaning 
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m 03_data_matching_v1
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m 04_data_organizing 
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m 05_nnUNet_labels_metrics_generation \
#     --input "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1/imagesTr" \
#     --output-dir "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1" \
#     --checkpoint-dir "/deepstore/datasets/mia/AI4EVAR/nnUNet/nnUNet_results/Dataset001_AAA_RADAR" \
#     --device "cuda"
srun -u $HOME/.conda/envs/ai4evar/bin/python -m 06_crop_full_body_scans \
    --input-dir /deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1 \
    --suffix _cropped #--dry-run
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m 07_generate_data_split 

echo $'\n--------------SLURM JOB FINISHED-----------------'
elapsed_time=$(($SECONDS - start_time))
echo "Elapsed time: $elapsed_time seconds"