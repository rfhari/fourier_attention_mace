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

    return atomic_energies_dict, A[0, :]

dimer_type = "CP"
dimer_ID = str(4)
print("dimer type and id:", dimer_ID, dimer_type)
kgrid_list = [4.0]
train_xyz = read(f"custom_dataset/dimer_datasets/vama_updated_dimer_{dimer_type}_{dimer_ID}_train.xyz", ":")
test_xyz = read(f"custom_dataset/dimer_datasets/vama_updated_dimer_{dimer_type}_{dimer_ID}_test.xyz", ":")
all_list = train_xyz + test_xyz

avge0, count_atom_types = compute_average_E0s(all_list, energy_key = "energy")
print("count_atom_types from ground truth:", count_atom_types)
ground_state = avge0[1] * count_atom_types[0] + avge0[6] * count_atom_types[1] + avge0[7] * count_atom_types[2] + avge0[8] * count_atom_types[3] 
print("ground state dft energies:", avge0, ground_state)

true_energy, true_forces, true_dipole_moment, distance_current = [], [], [], []

for i in range(len(all_list)):
    num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
    distance_current.append(all_list[i].info['distance'])
    true_dipole_moment.append(all_list[i].get_dipole_moment().reshape(3,))
    true_energy.append(all_list[i].get_potential_energy())
    true_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3,))

for kgrid in kgrid_list:
    # pred_energy, pred_forces = [], []  
    # calculator = MACECalculator(model_paths=f'./checkpoints/lr-mace-{dimer_type}-kgrid-{kgrid}-vama-6A_run-123_stagetwo.model', device='cpu')
    # mace_all_list = []

    # for i in range(len(all_list)):
    #     num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
    #     all_list[i].set_calculator(calculator)
    #     mace_all_list.append(all_list[i])
    #     pred_energy.append(all_list[i].get_potential_energy())  
    #     pred_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3))   

    sr_pred_energy, sr_pred_forces, sr_pred_dipoles = [], []    , []
    sr_calculator = MACECalculator(model_paths=f'./checkpoints/sr-DipoleMace-{dimer_type}-kgrid-{kgrid}-vama-6A_run-123_stagetwo.model', device='cuda')
    for i in range(len(all_list)):
        num_atoms_per_snapshot = len(all_list[i].get_atomic_numbers())
        all_list[i].set_calculator(sr_calculator)
        sr_pred_energy.append(all_list[i].get_potential_energy())
        # sr_pred_dipoles.append(all_list[i].get_dipole_moment()) 
        sr_pred_forces.append(all_list[i].get_forces().reshape(num_atoms_per_snapshot*3)) 

    np.save(f"sr_DipoleMace_{dimer_type}-kgrid-{kgrid}_pred_energy_6A.npy", sr_pred_energy)
    np.save(f"sr_DipoleMace_{dimer_type}-kgrid-{kgrid}_pred_force_6A.npy", sr_pred_forces)

    # sr_pred_energy = np.load(f"sr_pred_energy_6A_{dimer_type}_{dimer_ID}.npy")
    # sr_pred_forces = np.load(f"sr_pred_force_6A_{dimer_type}_{dimer_ID}.npy")

    plt.figure()
    # plt.plot(distance_current, pred_energy - ground_state, 'ro', markerfacecolor='none', label='lr predicted')
    plt.plot(distance_current, true_energy - ground_state, 'bo', label='DFT')
    plt.plot(distance_current, sr_pred_energy - ground_state, 'ko', markerfacecolor='none', label='sr predicted')
    plt.legend(frameon=False)
    plt.xlabel('distance (A)')
    plt.ylabel("binding energy (eV)")
    plt.savefig("dimer_"+dimer_type+"_"+dimer_ID+"_"+str(kgrid)+"_binding_curve.png", bbox_inches='tight')
    plt.show()

    print("finished plotting")