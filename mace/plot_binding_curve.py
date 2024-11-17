import ase 
from ase.io import write, read
from mace.calculators import MACECalculator
import matplotlib.pyplot as plt
import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Union, List
from ase import Atoms
import numpy as np


def get_unique_atomic_number(atoms_list: List[Atoms]) -> List[int]:
    unique_atomic_numbers = set()

    for atoms in atoms_list:
        unique_atomic_numbers.update(atom.number for atom in atoms)

    return list(unique_atomic_numbers)

def compute_average_E0s(
    atom_list: Atoms, zs = None, energy_key: str = "energy"
) -> Dict[int, float]:
    len_xyz = len(atom_list)
    if zs is None:
        zs = get_unique_atomic_number(atom_list)
        # sort by atomic number
        zs.sort()
    len_zs = len(zs)

    A = np.zeros((len_xyz, len_zs))
    B = np.zeros(len_xyz)
    for i in range(len_xyz):
        B[i] = atom_list[i].get_potential_energy()
        for j, z in enumerate(zs):
            A[i, j] = np.count_nonzero(atom_list[i].get_atomic_numbers() == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict

dimer_type = "CP"

train_xyz = read("custom_dataset/dimer_datasets/dimers_"+dimer_type+"_train.xyz", ":")
test_xyz = read("custom_dataset/dimer_datasets/dimers_"+dimer_type+"_test.xyz", ":")
all_list = train_xyz + test_xyz

# len(all_list)

avge0 = compute_average_E0s(all_list, energy_key = "energy")
true_energy, true_forces, distance_current = [], [], []

for i in range(len(all_list)):
    num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
    distance_current.append(all_list[i].info['distance'])
    true_energy.append(all_list[i].get_potential_energy())
    true_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3,))

# ground_state = avge0[1] * 10 + avge0[6] * 4 + avge0[7] * 2 + avge0[8] * 2
# print("ground state energies:", avge0)

pred_energy, pred_forces = [], []
calculator = MACECalculator(model_paths='./checkpoints/dimer-'+dimer_type+'-lr-remove-adjusted-mace_run-123_stagetwo.model', device='cuda')

for i in range(len(all_list)):
    num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
    # distance_current.append(all_list[i].info['distance'])
    # true_energy.append(all_list[i].get_potential_energy())
    # true_forces.append(all_list[i].get_forces().reshape(20*3,))
    all_list[i].set_calculator(calculator)
    pred_energy.append(all_list[i].get_potential_energy())  
    pred_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3))   

# sr_pred_energy = np.load("sr_pred_energy_6_A.npy")
# sr_pred_force = np.load("sr_pred_force_6_A.npy")

plt.figure()
plt.plot(distance_current, pred_energy, 'ro', label='lr predicted')
plt.plot(distance_current, true_energy, 'bo', label='true values')
# plt.plot(distance_current, sr_pred_energy-ground_state, 'ko', label='sr predicted')
plt.legend(frameon=False)
plt.xlabel('distance (A)')
plt.ylabel("binding energy (eV)")
plt.savefig("dimer_"+dimer_type+"_binding_curve_adjusted.png", bbox_inches='tight')
plt.show()

print("finished plotting")