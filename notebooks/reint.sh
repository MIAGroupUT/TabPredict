#!/bin/bash                 
#SBATCH -c 32               
#SBATCH --nodes=1       
#SBATCH --gres=gpu:1
#SBATCH --mem=180G          
#SBATCH --time=180:00:00     
#SBATCH --partition=main,mia,mia-pof,am
#SBATCH --output=/home/ranema/AI4EVAR/logs/%x_%j.log
#SBATCH --error=/home/ranema/AI4EVAR/logs/%x_%j.err   # stderr into same file

# 1. Enable the module command
source /etc/profile.d/modules.sh

# 2. Load Anaconda/Miniconda module (choose version available on UT HPC)
module load anaconda3/2025.06
export HTTPS_PROXY="http://proxy.utwente.nl:3128"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1

# cd $HOME/SIRE                   # Changes to your project directory
cd $HOME/AI4EVAR/ai4evar/notebooks

# 3. Run Python inside the conda environment WITHOUT activating it
start_time=$SECONDS
echo $'\n--------------SLURM JOB STARTED-----------------'

# srun -u $HOME/.conda/envs/ai4evar/bin/python -m "reintervention"
srun -u $HOME/.conda/envs/ai4evar/bin/python -m "reintervention_no_physics_no_complications"
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m "reintervention_with_complications"

echo $'\n--------------SLURM JOB FINISHED-----------------'
elapsed_time=$(($SECONDS - $start_time))
echo "Elapsed time: $elapsed_time"