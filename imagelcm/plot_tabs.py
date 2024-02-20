"""
Plots non-raster data output by IMAGE-LAND-LUE.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmap
from matplotlib.ticker import FormatStrFormatter

from read import check_wdir
import parameters as prm

# import xarray as xr

check_wdir()

crop_areas = np.load('outputs/crop_areas_5.npy')
reg_prod = np.load('outputs/reg_prod_5.npy')

food_dem = np.load('data/food_crop_demands.npy')
grass_dem = np.load('data/grass_crop_demands.npy')
food_dem = np.concatenate((grass_dem[:, 2, np.newaxis, :], food_dem), axis=1)

print(grass_dem.shape)

food_prod = np.swapaxes(reg_prod[:, 1:], 0, 1)
food_dem_1 = food_dem[1, :, :]

grass_prod = reg_prod[:, 0]
grass_mismatch = grass_dem[1, 2, :] - grass_prod

print(grass_prod.shape)

####### quick demand met test ########
# print(food_prod)
# print(food_dem_1)
# mismatch = food_dem[1, 1:, :] - food_prod
# mismatch[np.abs(mismatch)<prm.EPS] = 0.0
# print(mismatch)

print(grass_mismatch)
######################################

def load_tab(name, n_steps, invl, zero_start=False):
    """
    Loads and concatenates tabulated data for plotting of timeseries
    
    PARAMETERS
    ----------
    name : str
           name of the array to be loaded
    n_steps : int
              number of timesteps
    invl : int
           interval between timesteps
    zero_start : bool
                 whether to load initial data, or start from the first
                 model output
    """
    if zero_start:
        strt = 0
    else:
        strt = invl

    arrays = [np.load(f'outputs/{name}_{time}.npy') for time in range(strt, n_steps * invl, invl)]
    tab = np.stack(arrays, axis=0)

    return tab

def define_names():
    """
    Defines label names for all crop types and regions
    
    RETURNS
    -------
    crops : np.ndarray of type str
            array of crop type names
    regions : np.ndarray of type str
              array of region names
    """
    crops = ['grass', 'wheat', 'rice', 'maize', 'trop. cereals',
                        'oth. temp. cereals', 'pulses', 'soybeans', 'temp. oil crops',
                        'tropical oil crops', 'temp. roots+tubers', 'trop. roots+tubers',
                        'sugar crops', 'oil and palm fruit', 'veg+fruit',
                        'oth. non-food,\n luxury+spices', 'plant-based fibres']

    regions = ['CAN', 'US', 'MEX', 'R. CENT. AMER.', 'BR', 'R. SOU. AMER.',
                        'Northern Africa', 'Western Africa', 'Eastern Africa', 'South Africa',
                        'OECD Europe', 'Eastern Europe', 'Turkey', 'Ukraine +', 'Asia-Stan',
                        'Russia +', 'Middle East', 'India', 'Korea', 'China +', 'South East Asia',
                        'Indonesia +', 'Japan', 'Oceania', 'Rest of S. Asia', 'Rest of S. Africa']

    return crops, regions

def plot_timeseries_overview(tab, crops, regions, invl=None, title=None):
    """
    Plots a set of timeseries together by region and by crop 
    
    PARAMETERS
    ----------
    tab : np.ndarray of type np.float32
          3D array of shape (# timesteps, # crops, # regions)
    crops : np.ndarray of type str
            array of crop type names
    regions : np.ndarray of type str
              array of region names
    invl : int
           interval between timesteps
    """
    # define time array
    if invl is None:
        time = np.arange(tab.shape[0])
    else:
        time = np.arange(tab.shape[0]) * invl + 1970

    # by region
    fig, axes = plt.subplots(4, 7, sharex=True, sharey=True, figsize=(18, 9.5))
    axes = axes.flatten()

    for ind, ax in enumerate(axes):
        if ind==26:
            break
        for crop in range(16):
            ax.plot(time, tab[:, crop, ind], label=crops[crop])
        ax.set_title(regions[ind])

    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])

    if title is not None:
        fig.suptitle(title)

    axes[0].set_zorder(100)
    axes[0].legend(bbox_to_anchor=(8.3, -3.7), loc=4, ncols=2)

    # by crop
    fig2, axes2 = plt.subplots(4, 5, sharex=True, sharey=False, figsize=(18, 9.5))
    axes2 = axes2.flatten()

    for ind, ax in enumerate(axes2):
        if ind==17:
            break
        for reg in range(26):
            ax.plot(time, tab[:, ind, reg], label=regions[reg])
        ax.set_title(crops[ind])

    fig2.delaxes(axes2[-1])
    fig2.delaxes(axes2[-2])
    fig2.delaxes(axes2[-3])

    if title is not None:
        fig2.suptitle(title)

    axes2[0].set_zorder(100)
    axes2[0].legend(bbox_to_anchor=(5.5, -3.7), loc=4, ncols=4)

def plot_timeseries_separately(tab, crops, regions, invl=None, title=None):
    """
    Plots a set of timeseries separately by region and by crop
    """
    # define time array
    if invl is None:
        time = np.arange(tab.shape[0])
    else:
        time = np.arange(tab.shape[0]) * invl + 1970

    # by region
    for ind, _ in enumerate(regions):
        fig = plt.figure()
        for crop in range(17):
            plt.plot(time, tab[:, crop, ind], label=f'Crop {crop}')
        plt.title(f'{title} in {regions[ind]}')
        plt.legend(ncols=2)

def main():
    """main function"""

    crops, regs = define_names()
    plot_timeseries_overview(food_dem, crops, regs)

    crop_area_ts = load_tab('crop_areas', 5, 5, zero_start=True)

    # remove totals, combine rain-fed and irrigated and swap crop and region axes
    crop_area_ts = crop_area_ts[:, :-1, :-1]
    crop_area_ts[:, :, 1:prm.NGFC] += crop_area_ts[:, :, prm.NGFBC:]
    crop_area_ts = crop_area_ts[:, :, :prm.NGFC]
    crop_area_ts = np.swapaxes(crop_area_ts, 1, 2)

    # divide by a million
    crop_area_ts /= 1e6

    # print(crop_area_ts[1, 0, :])

    # make a fraction of initial values
    # crop_area_ts /= crop_area_ts[0, :, :]

    print(crop_area_ts.shape)

    plot_timeseries_overview(crop_area_ts, crops, regs, invl=5, title='Crop Area [m. km^2]')

    plot_timeseries_separately(crop_area_ts, crops, regs, invl=5, title='Cop Area [m. km^2]')

    plt.show()

if __name__=='__main__':
    main()
