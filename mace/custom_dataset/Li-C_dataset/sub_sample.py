import ase 
from ase.io import read, write
import numpy as np

train_atoms = read("Li-C-888-train.xyz", ":")
test_atoms = read("Li-C-888-test.xyz", ":")

print(len(train_atoms), len(test_atoms))

np.random.shuffle(train_atoms)
np.random.shuffle(test_atoms)

write("Li-C-888-trimmed-train.xyz", train_atoms[:650])
write("Li-C-888-trimmed-test.xyz", test_atoms[:100])

print(len(train_atoms), len(test_atoms))
