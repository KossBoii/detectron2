#!/bin/bash
#SBATCH --job-name=road_stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o ./slurm_log/output_%j.txt
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=lntruong@cpp.edu

eval "$(conda shell.bash hook)"
conda activate py3
srun python ./train_net.py 
