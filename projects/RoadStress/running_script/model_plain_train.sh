#!/bin/bash
#SBATCH --job-name=road_stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -o ./slurm_log/output_%j.txt
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=lntruong@cpp.edu

eval "$(conda shell.bash hook)"
conda activate py3
dir=`pwd`
echo "$dir/roadstress_new"
if [ ! -d "$dir/roadstress_new" ]; then
	svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_new
else
	echo "Dataset exists"
fi
srun python ./train_script.py train --weights=coco --dataset=./roadstress_new/ --num-gpus 2 
