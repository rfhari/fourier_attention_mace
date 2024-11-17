#!/bin/bash 

#SBATCH -J mace_exp
#SBATCH -p batch
#SBATCH -o mace_lr_%j.out
#SBATCH --time=10:00:00
#SBATCH -N 1           
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gpus=1

MINICONDA_PATH=/trace/group/mcgaughey/hariharr/miniconda3
source $MINICONDA_PATH/etc/profile.d/conda.sh

conda activate /trace/group/mcgaughey/hariharr/miniconda3/envs/mace

export PYTHONPATH=/trace/group/mcgaughey/hariharr/mace_exploration/fourier_attention_mace/mace_mha:$PYTHONPATH

# python /trace/group/mcgaughey/hariharr/mace_exploration/fourier_attention_mace/mace/mace/cli/run_train.py \
#     --log_dir="./logs" \
#     --energy_key="energy" \
#     --forces_key="forces" \
#     --name="dimer-cc-lr-mace-vslides" \
#     --train_file="./custom_dataset/dimer_cc/dimers_cc_train.xyz" \
#     --valid_file="./custom_dataset/dimer_cc/dimers_cc_test.xyz" \
#     --test_file="./custom_dataset/dimer_cc/dimers_cc_test.xyz" \
#     --E0s="average" \
#     --model="MACE" \
#     --num_interactions=2 \
#     --num_channels=256 \
#     --max_L=2 \
#     --correlation=3 \
#     --r_max=6 \
#     --forces_weight=1000 \
#     --energy_weight=10 \
#     --batch_size=2 \
#     --valid_batch_size=2 \
#     --max_num_epochs=2000 \
#     --start_swa=1200 \
#     --scheduler_patience=5 \
#     --patience=15 \
#     --eval_interval=3 \
#     --ema \
#     --swa \
#     --swa_forces_weight=1000 \
#     --error_table='PerAtomMAE' \
#     --default_dtype="float64"\
#     --device=cuda \
#     --seed=123 \
#     --save_cpu


python ./mace/cli/run_train.py \
    --log_dir="./logs" \
    --energy_key="energy" \
    --forces_key="forces" \
    --name="dimer-CP-lr-remove-adjusted-mace" \
    --train_file="./custom_dataset/dimer_datasets/dimers_CP_train.xyz" \
    --valid_file="./custom_dataset/dimer_datasets/dimers_CP_test.xyz" \
    --test_file="./custom_dataset/dimer_datasets/dimers_CP_test.xyz" \
    --E0s="average" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=256 \
    --max_L=2 \
    --correlation=3 \
    --r_max=6 \
    --forces_weight=100 \
    --energy_weight=10 \
    --batch_size=2 \
    --valid_batch_size=2 \
    --max_num_epochs=800 \
    --start_swa=600 \
    --scheduler_patience=5 \
    --patience=5 \
    --eval_interval=3 \
    --ema \
    --swa \
    --lr=0.01 \
    --swa_forces_weight=10 \
    --swa_energy_weight=100 \
    --error_table='PerAtomMAE' \
    --default_dtype="float64"\
    --device=cuda \
    --seed=123 

# python mace/cli/run_train.py \
#     --name="MACE" \
#     --train_file="/home/hari/Desktop/Research/mace_fourier_attention/mace/custom_dataset/buckyball_catcher/buckyball_catcher_train_eV.xyz" \
#     --valid_file="/home/hari/Desktop/Research/mace_fourier_attention/mace/custom_dataset/buckyball_catcher/buckyball_catcher_test_eV.xyz" \
#     --test_file="/home/hari/Desktop/Research/mace_fourier_attention/mace/custom_dataset/buckyball_catcher/buckyball_catcher_test_eV.xyz" \
#     --config_type_weights='{"Default":1.0}' \
#     --E0s='average' \
#     --model="MACE" \
#     --hidden_irreps='128x0e + 128x1o' \
#     --r_max=5.0 \
#     --batch_size=10 \
#     --max_num_epochs=100 \
#     --swa \
#     --start_swa=90 \
#     --ema \
#     --ema_decay=0.99 \
#     --amsgrad \
#     --device=cpu \
#     --save_cpu


# python /home/hari/Desktop/Research/mace_fourier_attention/mace/mace/cli/run_train.py \
#     --name="buckyball" \
#     --train_file="nanotube_large.xyz" \
#     --valid_fraction=0.05 \
#     --test_file="nanotube_test.xyz" \
#     --E0s="average" \
#     --model="MACE" \
#     --num_interactions=1 \
#     --num_channels=256 \
#     --max_L=0 \
#     --correlation=3 \
#     --r_max=6.0 \
#     --forces_weight=1000 \
#     --energy_weight=10 \
#     --batch_size=4 \
#     --valid_batch_size=8 \
#     --max_num_epochs=1000 \
#     --start_swa=600 \
#     --scheduler_patience=5 \
#     --patience=15 \
#     --eval_interval=3 \
#     --ema \
#     --swa \
#     --swa_forces_weight=10 \
#     --error_table='PerAtomMAE' \
#     --default_dtype="float64"\
#     --device=cuda \
#     --seed=123 \
#     --restart_latest \
#     --save_cpu



# python /grand/QuantumDS/hariharr/MACE_LR/mace_train/scripts/run_train.py \
#     --log_dir='/grand/QuantumDS/hariharr/MACE_LR/mace_train/logs'\
#     --checkpoints_dir='/grand/QuantumDS/hariharr/MACE_LR/mace_train/checkpoints' \
#     --use_pbc \
#     --name="MACE_Ewald_NaCl_water_test" \
#     --num_workers=16 \
#     --energy_key="energy" \
#     --forces_key="forces" \
#     --r_max=3.0 \
#     --train_file="/grand/QuantumDS/hariharr/MACE_LR/mace_train/custom_data/nao2_water/nacl_train_data_eV.xyz" \
#     --valid_fraction=0.05 \
#     --test_file="/grand/QuantumDS/hariharr/MACE_LR/mace_train/custom_data/nao2_water/nacl_test_data_eV.xyz" \
#     --model="MACE_Ewald" \
#     --num_interactions=2 \
#     --num_channels=256 \
#     --E0s="average" \
#     --max_L=2 \
#     --correlation=3 \
#     --batch_size=2 \
#     --valid_batch_size=2 \
#     --max_num_epochs=10 \
#     --swa \
#     --start_swa=5 \
#     --ema \
#     --ema_decay=0.99 \
#     --amsgrad \
#     --eval_interval=3 \
#     --forces_weight=1000 \
#     --energy_weight=10 \
#     --scheduler_patience=5 \
#     --patience=15 \
#     --swa_forces_weight=10 \
#     --error_table='PerAtomMAE' \
#     --device=cuda \
#     --default_dtype="float64" \
#     --seed=456 \
#     --restart_latest \
#     --save_cpu > /grand/QuantumDS/hariharr/MACE_LR/mace_train/nacl_water_key_test.out    