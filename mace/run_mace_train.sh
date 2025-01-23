#!/bin/bash 

#SBATCH -J li-graphene
#SBATCH -p batch
#SBATCH -o mace_lr_%j.out
#SBATCH --time=47:00:00
#SBATCH -N 1           
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=32
## SBATCH --gpus=1

MINICONDA_PATH=/trace/group/mcgaughey/hariharr/miniconda3
source $MINICONDA_PATH/etc/profile.d/conda.sh

conda activate /trace/group/mcgaughey/hariharr/miniconda3/envs/mace

export PYTHONPATH=/trace/group/mcgaughey/hariharr/mace_exploration/fourier_attention_mace/mace:$PYTHONPATH

# ------------------------------------- Li-C ----------------------------
# dimer_type=Li-C
# supercell=888
# k_grid=4.0

# kernprof -l ./mace/cli/run_train.py \
#     --log_dir="./logs" \
#     --energy_key="energy" \
#     --forces_key="forces" \
#     --name="profile-lr-mace-${dimer_type}-${supercell}-kgrid-${k_grid}-6A" \
#     --train_file="./custom_dataset/Li-C_dataset/${dimer_type}-${supercell}-trimmed-train.xyz" \
#     --valid_file="./custom_dataset/Li-C_dataset/${dimer_type}-${supercell}-trimmed-test.xyz" \
#     --test_file="./custom_dataset/Li-C_dataset/${dimer_type}-${supercell}-trimmed-test.xyz" \
#     --E0s="average" \
#     --model="MACE" \
#     --num_interactions=1 \
#     --num_channels=256 \
#     --max_L=1 \
#     --correlation=1 \
#     --r_max=6 \
#     --forces_weight=1000 \
#     --energy_weight=10 \
#     --batch_size=2 \
#     --valid_batch_size=8 \
#     --max_num_epochs=10 \
#     --start_swa=6 \
#     --scheduler_patience=5 \
#     --patience=5 \
#     --eval_interval=3 \
#     --ema \
#     --swa \
#     --lr=0.01 \
#     --swa_forces_weight=10 \
#     --swa_energy_weight=1000 \
#     --error_table='PerAtomMAE' \
#     --default_dtype="float64"\
#     --device=cpu \
#     --seed=123 \

# ------------------------------------- different dimers ----------------------------
dimer_type=PP
dimer_id=5
k_grid=5.0

kernprof -l ./mace/cli/run_train.py \
    --log_dir="./logs" \
    --energy_key="energy" \
    --forces_key="forces" \
    --name="profile-lr-mace-${dimer_type}-kgrid-${k_grid}-vama-6A" \
    --train_file="./custom_dataset/dimer_datasets/vama_updated_dimer_${dimer_type}_${dimer_id}_train.xyz" \
    --valid_file="./custom_dataset/dimer_datasets/vama_updated_dimer_${dimer_type}_${dimer_id}_test.xyz" \
    --test_file="./custom_dataset/dimer_datasets/vama_updated_dimer_${dimer_type}_${dimer_id}_test.xyz" \
    --E0s="average" \
    --model="MACE" \
    --num_interactions=1 \
    --num_channels=256 \
    --max_L=1 \
    --correlation=1 \
    --r_max=6 \
    --forces_weight=1000 \
    --energy_weight=10 \
    --batch_size=2 \
    --valid_batch_size=8 \
    --max_num_epochs=6 \
    --start_swa=3 \
    --scheduler_patience=5 \
    --patience=5 \
    --eval_interval=3 \
    --ema \
    --swa \
    --lr=0.01 \
    --swa_forces_weight=10 \
    --swa_energy_weight=1000 \
    --error_table='PerAtomMAE' \
    --default_dtype="float64"\
    --device=cpu \
    --seed=123 \
    --restart_latest

# ------------------------------------- water clusters ----------------------------
# config_id=1
# python ./mace/cli/run_train.py \
#     --log_dir="./logs" \
#     --energy_key="energy" \
#     --forces_key="forces" \
#     --name="lr-mace-water-${config_id}-6A" \
#     --train_file="./custom_dataset/water_clusters/combined_dimer_test_config_${config_id}_train.xyz" \
#     --valid_file="./custom_dataset/water_clusters/combined_dimer_test_config_${config_id}_test.xyz" \
#     --test_file="./custom_dataset/water_clusters/combined_dimer_test_config_${config_id}_test.xyz" \
#     --E0s="average" \
#     --model="MACE" \
#     --num_interactions=2 \
#     --num_channels=256 \
#     --max_L=1 \
#     --correlation=1 \
#     --r_max=6 \
#     --forces_weight=1000 \
#     --energy_weight=10 \
#     --batch_size=16 \
#     --valid_batch_size=16 \
#     --max_num_epochs=295 \
#     --start_swa=192 \
#     --scheduler_patience=5 \
#     --patience=5 \
#     --eval_interval=3 \
#     --ema \
#     --swa \
#     --lr=0.01 \
#     --swa_forces_weight=10 \
#     --swa_energy_weight=1000 \
#     --error_table='PerAtomMAE' \
#     --default_dtype="float64"\
#     --device=cpu \
#     --seed=123 \
#     --restart_latest

# ------------------------------------- buckyball ----------------------------
# python ./mace/cli/run_train.py \
#     --log_dir="./logs" \
#     --name="bucky_ball_3_3" \
#     --energy_key="energy" \
#     --forces_key="forces" \
#     --train_file="./custom_dataset/buckyball_catcher/buckyball_catcher_train_eV_periodic.xyz" \
#     --valid_file="./custom_dataset/buckyball_catcher/buckyball_catcher_test_eV_periodic.xyz" \
#     --test_file="./custom_dataset/buckyball_catcher/buckyball_catcher_test_eV_periodic.xyz" \
#     --valid_fraction=0.05 \
#     --E0s="average" \
#     --model="MACE" \
#     --num_interactions=2 \
#     --num_channels=256 \
#     --max_L=0 \
#     --correlation=1 \
#     --r_max=3.0 \
#     --forces_weight=1000 \
#     --energy_weight=10 \
#     --batch_size=2 \
#     --valid_batch_size=2 \
#     --max_num_epochs=1000 \
#     --start_swa=700 \
#     --scheduler_patience=5 \
#     --patience=15 \
#     --eval_interval=3 \
#     --ema \
#     --swa \
#     --lr=0.01 \
#     --swa_forces_weight=10 \
#     --swa_energy_weight=1000 \
#     --error_table='PerAtomMAE' \
#     --default_dtype="float64"\
#     --device=cuda \
#     --seed=123 \