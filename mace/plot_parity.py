import ase 
from ase.io import write, read
from mace.calculators import MACECalculator
import matplotlib.pyplot as plt
import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Union, List
from ase import Atoms
import numpy as np

config_id = 1
distance = np.arange(0.5, 15.1, 0.1)

train_xyz = read(f"./custom_dataset/water_clusters/combined_dimer_test_config_{config_id}_train.xyz", ":")
test_xyz = read(f"./custom_dataset/water_clusters/combined_dimer_test_config_{config_id}_test.xyz", ":")
all_list = train_xyz + test_xyz

true_energy, true_forces, distance_current = [], [], []

for i in range(len(all_list)):
    num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
    true_energy.append(all_list[i].get_potential_energy())
    true_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3,))

pred_energy, pred_forces = [], []  
calculator = MACECalculator(model_paths=f'./checkpoints/lr-mace-water-{config_id}-6A_run-123_stagetwo.model', device='cpu')
mace_all_list = []

for i in range(len(all_list)):
    num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
    all_list[i].set_calculator(calculator)
    pred_energy.append(all_list[i].get_potential_energy())  
    pred_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3))   

plt.figure()
plt.plot(distance, pred_energy, 'ro', markerfacecolor='none', label='lr predicted')
plt.plot(distance, true_energy, 'bo', label='true values')
# plt.plot(distance_current, sr_pred_energy, 'ko', markerfacecolor='none', label='sr predicted')
plt.legend(frameon=False)
plt.xlabel('distance (A)')
plt.ylabel("binding energy (eV)")
plt.savefig(f"water_{config_id}_binding_curve.png", bbox_inches='tight')
plt.show()

print("finished plotting")