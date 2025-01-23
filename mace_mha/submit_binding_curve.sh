#!/bin/bash 

#SBATCH -J mace_exp
#SBATCH -p mcgaughey
#SBATCH -o mace_lr_%j.out
#SBATCH --time=00:15:00
#SBATCH -N 1           
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gpus=1

MINICONDA_PATH=/trace/group/mcgaughey/hariharr/miniconda3
source $MINICONDA_PATH/etc/profile.d/conda.sh

conda activate /trace/group/mcgaughey/hariharr/miniconda3/envs/mace

export PYTHONPATH=/trace/group/mcgaughey/hariharr/mace_exploration/fourier_attention_mace/mace:$PYTHONPATH

python plot_binding_curve.py

