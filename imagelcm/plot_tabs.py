"""
Plots non-raster data output by IMAGE-LAND-LUE.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmap
from read import check_wdir
import parameters as prm

# import xarray as xr

check_wdir()

crop_areas = np.load('outputs\\crop_areas_1.npy')
reg_prod = np.load('outputs\\reg_prod_1.npy')

food_dem = np.load('data\\food_crop_demands.npy')
grass_dem = np.load('data\\grass_crop_demands.npy')

food_prod = np.swapaxes(reg_prod[:, 1:], 0, 1)
food_dem_1 = food_dem[1, :, :]

####### quick demand met test ########
# print(food_prod)
print(food_dem_1)
mismatch = food_dem[1, :, :] - food_prod
mismatch[np.abs(mismatch)<prm.EPS] = 0.0
print(mismatch)
######################################

time = np.arange(131)

ratio_1 = food_dem_1 / food_dem[0, :, :]

crops = np.asarray(['grass', 'wheat', 'rice', 'maize', 'trop. cereals',
                    'oth. temp. cereals', 'pulses', 'soybeans', 'temp. oil crops',
                    'tropical oil crops', 'temp. roots+tubers', 'trop. roots+tubers',
                    'sugar crops', 'oil and palm fruit', 'veg+fruit',
                    'oth. non-food,\n luxury+spices', 'plant-based fibres'])

regions = np.asarray(['CAN', 'US', 'MEX', 'R. CENT. AMER.', 'BR', 'R. SOU. AMER.',
                      'Northern Africa', 'Western Africa', 'Eastern Africa', 'South Africa',
                      'OECD Europe', 'Eastern Europe', 'Turkey', 'Ukraine +', 'Asia-Stan',
                      'Russia +', 'Middle East', 'India', 'Korea', 'China +', 'South East Asia',
                      'Indonesia +', 'Japan', 'Oceania', 'Rest of S. Asia', 'Rest of S. Africa'])

# print(food_prod - food_dem_1)

fig, axes = plt.subplots(4, 7, sharex=True, sharey=True, figsize=(18, 9.5))
axes = axes.flatten()

colors = cmap['turbo'](np.linspace(0, 1, 16))

for ind, ax in enumerate(axes):
    if ind==26:
        break
    for crop in range(16):
        ax.plot(time, food_dem[:, crop, ind], label=crops[crop+1])
    # if ind==0:
    #     ax.legend()
    ax.set_title(regions[ind])

fig.delaxes(axes[-1])
fig.delaxes(axes[-2])

axes[0].set_zorder(100)
axes[0].legend(bbox_to_anchor=(8.3, -3.7), loc=4, ncols=2)

fig.tight_layout()

plt.show()
