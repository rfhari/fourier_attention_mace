#!/bin/bash 

source /home/hari/anaconda3/etc/profile.d/conda.sh
conda activate /home/hari/anaconda3/envs/mace_fr

export PYTHONPATH=/home/hari/Desktop/Research/mace_fourier_attention/mace:$PYTHONPATH

python /home/hari/Desktop/Research/mace_fourier_attention/mace/mace/cli/run_train.py \
    --log_dir="/home/hari/Desktop/Research/mace_fourier_attention/mace/logs" \
    --name="ca-water-periodic" \
    --train_file="/home/hari/Desktop/Research/mace_fourier_attention/mace/custom_dataset/cawaterpbc/cawater_train.xyz" \
    --valid_fraction=0.05 \
    --test_file="/home/hari/Desktop/Research/mace_fourier_attention/mace/custom_dataset/cawaterpbc/cawater_test.xyz" \
    --E0s="average" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=256 \
    --max_L=0 \
    --correlation=3 \
    --r_max=6.0 \
    --forces_weight=1000 \
    --energy_weight=10 \
    --batch_size=4 \
    --valid_batch_size=8 \
    --max_num_epochs=10 \
    --start_swa=6 \
    --scheduler_patience=5 \
    --patience=5 \
    --eval_interval=3 \
    --ema \
    --swa \
    --swa_forces_weight=10 \
    --error_table='PerAtomMAE' \
    --default_dtype="float64"\
    --device=cpu \
    --seed=123 \
    --save_cpu


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