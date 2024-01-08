import numpy as np
from read import check_wdir
import parameters as prm

check_wdir()

MFs = np.load('data\\MF.npy')
FHs = np.load('data\\FH.npy')

print(MFs.shape)
# print(MFs[0, :prm.NFC, :])

R1_integration = np.load('outputs\\regional_prods_R1.npy')
print(R1_integration)

# print(R1_integration)

# print(FHs.shape)
# print(FHs)
