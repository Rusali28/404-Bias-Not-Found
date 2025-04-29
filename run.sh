#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=genmatrix_param
#SBATCH --output=logs/job.o%J
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-user=rs0921@tamu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu   #Request job to be put in the GPU queue
#SBATCH --gres=gpu:2      #Request 1 GPU per node can be 1 or 2

## Load the module

module load GCCcore/10.3.0
module load Python/3.9.5
module load CUDA/12.4.1
# module load GCCcore/10.3.0
#  module spider Python/3.9.5
#  module load Python/3.9.5


# === Job Info ===
echo "Running job on $(hostname) at $(date)"

# === Load Environment ===
source /scratch/user/rs0921/ISR-PAAC-project/isr_venv2/bin/activate

# === Run Python Script ===
python -u PAAC_main.py --dataset_name yelp2018 --layers_list '[5]' --cl_rate_list '[10]' --align_reg_list '[1e3]' --lambada_list '[0.3]' --gama_list '[0.8]' --num_epoch 15

#running algorithm on gowalla
# python -u PAAC_main.py --dataset_name gowalla --layers_list '[6]' --cl_rate_list '[5]' --align_reg_list '[50]' --lambada_list '[0.2]' --gama_list '[0.2]' --num_epoch 30

echo "Job finished at $(date)"