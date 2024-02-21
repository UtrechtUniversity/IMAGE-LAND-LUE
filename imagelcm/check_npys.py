import numpy as np
from read import check_wdir
import parameters as prm

# import xarray as xr

check_wdir()

crops = np.asarray(['grass', 'wheat', 'rice', 'maize', 'tropical cereals',
                    'other temperate cereals', 'pulses', 'soybeans', 'temperate oil crops',
                    'tropical oil crops', 'temperate roots n tubers', 'tropical roots n tubers',
                    'sugar crops', 'oil and palm fruit', 'veg n fruit', 
                    'other non-food, luxury n spices', 'plant-based fibres'])

regions = np.asarray(['CAN', 'US', 'MEX', 'R. CENT. AMER.', 'BR', 'R. SOU. AMER.',
                      'Northern Africa', 'Western Africa', 'Eastern Africa', 'South Africa',
                      'OECD Europe', 'Eastern Europe', 'Turkey', 'Ukraine +', 'Asia-Stan',
                      'Russia +', 'Middle East', 'India', 'Korea', 'China +', 'South East Asia',
                      'Indonesia +', 'Japan', 'Oceania', 'Rest of S. Asia', 'Rest of S. Africa'])

MFs = np.load('data/MF.npy')
FHs = np.load('data/FH.npy')
GIs = np.load('data/GI.npy')

# print(MFs.shape)
# print(MFs[0, :prm.NFC, :])

print(GIs.shape)
print(GIs[0, :, :])

# R1_integration = np.load('outputs/regional_prods_R1.npy')
# print(R1_integration)

# irr_yield = np.load('outputs/ir_yields.npy')

# print(irr_yield[17, :])
# print(regions[np.argmax(irr_yield, axis=0)])
# print(np.max(irr_yield, axis=0))

# greg = xr.open_dataarray('data/GREG_5MIN.nc')
# greg = greg.fillna(0)
# greg = greg.astype(np.int32)

# greg.to_netcdf('data/greg_5min_int.nc')

# print(R1_integration)

# print(FHs.shape)
# print(FHs)
