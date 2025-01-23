import ase 
from ase.io import write, read
from mace.calculators import MACECalculator
import matplotlib.pyplot as plt
import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Union, List
from ase import Atoms
import numpy as np
import time

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
        count_atom_types = {}
        for i, z in enumerate(zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(zs):
            atomic_energies_dict[z] = 0.0

    return atomic_energies_dict, A[:, :]

start_time = time.time()
dimer_type = "Li-C"
size_supercell = "888"
print("dimer type and id:", dimer_type, size_supercell)

train_xyz = read("custom_dataset/Li-C_dataset/Li-C-"+size_supercell+"-trimmed-train.xyz", ":")
test_xyz = read("custom_dataset/Li-C_dataset/Li-C-"+size_supercell+"-trimmed-test.xyz", ":")

all_list = train_xyz + test_xyz

avge0, count_atom_types = compute_average_E0s(all_list, energy_key = "energy")
print("count_atom_types from ground truth:", count_atom_types)

ground_state = avge0[3] * count_atom_types[:, 0] + avge0[6] * count_atom_types[:, 1]
print("ground state dft energies:", avge0, ground_state)

true_energy, true_forces = [], []

for i in range(len(all_list)):
    num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
    true_energy.append(all_list[i].get_potential_energy())
    true_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3,))

# sr_energies = np.load(f"sr_energies_{dimer_type}_{size_supercell}.npy")

all_distances, pred_energy, pred_forces, li_dimer_ind, num_atoms = [], [], [], [], []
calculator = MACECalculator(model_paths='./checkpoints/lr-mace-Li-C-'+'444-trimmed'+'-smaller_run-123_stagetwo.model', device='cuda')

for i, snapshot in enumerate(all_list):
    li_indices = [j for j, atom in enumerate(snapshot) if atom.symbol == 'Li']
    if len(li_indices) == 2:
        pos1 = snapshot[li_indices[0]].position
        pos2 = snapshot[li_indices[1]].position
        dist = np.sqrt(np.sum(np.square(pos1 - pos2))) 
        print(dist)
        all_distances.append(dist)
        num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
        num_atoms.append(num_atoms_per_snapshot)
        all_list[i].set_calculator(calculator)
        pred_energy.append(all_list[i].get_potential_energy())  
        # pred_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3))   
        li_dimer_ind.append(i)

# all_distances, sr_pred_energy, sr_pred_forces, li_dimer_ind, num_atoms = [], [], [], [], []
# sr_calculator = MACECalculator(model_paths='./checkpoints/sr-mace-Li-C-'+'444-trimmed'+'-smaller_run-123_stagetwo.model', device='cuda')

# for i, snapshot in enumerate(all_list):
#     li_indices = [j for j, atom in enumerate(snapshot) if atom.symbol == 'Li']
#     if len(li_indices) == 2:
#         pos1 = snapshot[li_indices[0]].position
#         pos2 = snapshot[li_indices[1]].position
#         dist = np.sqrt(np.sum(np.square(pos1 - pos2))) 
#         print(dist)
#         all_distances.append(dist)
#         num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
#         num_atoms.append(num_atoms_per_snapshot)
#         all_list[i].set_calculator(sr_calculator)
#         sr_pred_energy.append(all_list[i].get_potential_energy())  
#         # sr_pred_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3))   
#         li_dimer_ind.append(i)

# np.save(f"sr_energies_{dimer_type}_{size_supercell}.npy", (sr_pred_energy - ground_state[li_dimer_ind]))
# np.save(f"lr_energies_{dimer_type}_{size_supercell}.npy", (pred_energy[:] - ground_state[li_dimer_ind]))
# np.save(f"DFT_energies_{dimer_type}_{size_supercell}.npy", (np.asarray(true_energy)[li_dimer_ind] - ground_state[li_dimer_ind]))
sr_pred_energy = np.load(f"sr_energies_{dimer_type}_{size_supercell}.npy") + ground_state[li_dimer_ind]

total_num_atoms = 130

plt.figure()
plt.plot(all_distances, (np.asarray(true_energy)[li_dimer_ind] - ground_state[li_dimer_ind])/total_num_atoms, 'bo', label='DFT')
plt.plot(all_distances, (pred_energy[:] - ground_state[li_dimer_ind])/total_num_atoms, 'g*', label='LR-model')
# plt.plot(all_distances, (sr1_pred_energy - ground_state[li_dimer_ind])/total_num_atoms, 'g*', label='SR-model')
plt.plot(all_distances, (sr_pred_energy - ground_state[li_dimer_ind])/total_num_atoms, 'rx', label='SR-smaller-model')
plt.legend(frameon=True)
plt.ylabel('binding energy (eV)')
plt.xlabel("Li-Li distance")
plt.savefig("Li-C-comparison-"+size_supercell+".png", dpi=300, bbox_inches='tight')
plt.show()

close_time = time.time()
print("finished plotting")
print("total time taken (sec):", close_time - start_time)