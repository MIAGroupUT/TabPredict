#!/bin/bash                 
#SBATCH -c 4   
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --mem=180G          
#SBATCH --time=370:00:00     
#SBATCH --partition=mia,mia-pof,am,main
#SBATCH --output=/home/ranema/AI4EVAR/logs/%x_%j.log
#SBATCH --error=/home/ranema/AI4EVAR/logs/%x_%j.err   # stderr into same file

# 1. Enable the module command
source /etc/profile.d/modules.sh

# 2. Load Anaconda/Miniconda module (choose version available on UT HPC)
module load anaconda3/2025.06
source activate ai4evar

export HTTPS_PROXY="http://proxy.utwente.nl:3128"

cd $HOME/AI4EVAR                 # Changes to your project directory

# 3. Run Python inside the conda environment WITHOUT activating it
start_time=$SECONDS
echo $'\n--------------SLURM JOB STARTED-----------------'

# srun -u $HOME/.conda/envs/ai4evar/bin/python -m prepare.test_mistral_download
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m prepare.spss_to_json_filter_Irene --spss /deepstore/datasets/mia/AI4EVAR/clinical_data/SPSS_Irene_RADAR.sav --output /deepstore/datasets/mia/AI4EVAR/clinical_data/SPSS_Irene_RADAR_filtered.json
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m prepare.spss_to_json_grouped_using_LLM
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m prepare.generate_reports --input /deepstore/datasets/mia/AI4EVAR/clinical_data/SPSS_Irene_RADAR_filtered.json --patient "0119"
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m prepare.generate_reports --input /deepstore/datasets/mia/AI4EVAR/clinical_data/SPSS_Irene_RADAR_filtered.json --patient "0141"
# srun -u $HOME/.conda/envs/ai4evar/bin/python -m prepare.generate_reports --input /deepstore/datasets/mia/AI4EVAR/clinical_data/SPSS_Irene_RADAR_filtered.json

echo $'\n--------------SLURM JOB FINISHED-----------------'
elapsed_time=$(($SECONDS - $start_time))
echo "Elapsed time: $elapsed_time"