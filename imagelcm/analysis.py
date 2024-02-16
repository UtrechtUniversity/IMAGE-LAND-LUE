"""
Functions for the analysis of crop fractions allocated in landalloc.py.
"""
import numpy as np
import lue.framework as lfr

import parameters as prm
import read as rd
import write as wt

def compute_largest_fraction(fracs, is_cropland, save=False, timestep=0):
    """
    Finds the largest crop fraction and the crop it belongs to

    Parameters
    ----------
    fracs : list of LUE data objects
            list of arrays containing the crop fractions for all cells
    save : bool, default = False
           if true, save outputs in netcdf format

    Returns
    -------
    ma_frac : LUE data object
              2D array containing the largest crop fraction, for all cells
    ma_crop : LUE data object
              2D array containing the most allocated crop, for all cells
    """

    f_len = len(fracs)

    # set initial values to 1st crop('s fractions)
    ma_frac = lfr.where(is_cropland, 0., -1.)
    ma_crop = lfr.where(is_cropland, 0, -1)

    # update by iterating through all crops
    for crop in range(0, min(prm.NGFC, f_len)):
        current_crop_bigger = (fracs[crop] > ma_frac) & is_cropland
        ma_frac = lfr.where(current_crop_bigger, fracs[crop], ma_frac)
        ma_crop = lfr.where(current_crop_bigger, crop, ma_crop)

    if save:
        wt.write_raster(ma_frac, f'mafrac_{timestep}')
        wt.write_raster(ma_crop, f'mac{timestep}')

    return ma_frac, ma_crop

def compute_crop_areas(fracs, garea, regions):
    """
    Computes the total area allocated to cropland and each crop
    
    PARAMETERS
    ----------
    fracs : list
            list of lue PartitionedArrays containing the most recent crop
            fractions
    garea : lue PartitionedArray<>
            array showing the amount of area in each grid cell
    regions : lue PartitionedArray<>
              array containing the region to which each grid cell belongs
    
    RETURNS
    -------
    crop_areas : np.ndarray
                 2-dimensional array containing the area of land allocated
                 to each crop (and agriculture in total) in each region
                 and globally
    """

    regional_areas = [lfr.zonal_sum(frac*garea, regions) for frac in fracs]

    crop_areas = np.zeros((prm.N_REG+1, prm.NGFBFC+1))
    for ind, reg_ar in enumerate(regional_areas):
        # isolate crop_areas for each region
        u, indices = np.unique(lfr.to_numpy(reg_ar), return_index=True)
        corresponding_regions = lfr.to_numpy(regions).flatten()[indices]

        # make sure that regions 0 and 27 are removed
        where_valid_regions = np.logical_and(corresponding_regions>0,
                                             corresponding_regions<=prm.N_REG)
        u = u[where_valid_regions]
        corresponding_regions = corresponding_regions[where_valid_regions]

        crop_areas[corresponding_regions-1, ind] = u

    # include regional/agricultural totals
    crop_areas[-1, :-1] = np.sum(crop_areas[:, :-1], axis=0)
    crop_areas[-1, -1] = crop_areas[-1, :-1].sum()

    return crop_areas

def compute_diff_rasters(fracs_1, fracs_2, timestep_1, timestep_2, save=False):
    """
    Computes the difference in crop fractions from one timestep to another

    PARAMETERS
    ----------
    fracs_1 : list
              list of lue.PartitionedArray<float32> containing the initial
              crop fractions
    fracs_2 : list
              list of lue.PartitionedArray<float32> containing the updated
              crop fractions
    timestep_1 : int
                 initial timestep
    timestep_2 : int
                 timestep of updated crop fractions
    save : bool, default = False
           whether to save the resulting difference rasters
    
    RETURNS
    -------
    diff_rasters : list
                   list of lue.PartitionedArray<float32> containing the
                   change in crop fractions
    """
    diff_rasters = []

    print(f'fracs1 length: {len(fracs_1)}')
    print(f'fracs2 length: {len(fracs_2)}')

    for crop in range(len(fracs_1)):
        diff_rasters.append(fracs_2[crop] - fracs_1[crop])

    if save:
        rd.check_wdir()
        for crop in range(len(diff_rasters)):
            wt.write_raster(diff_rasters[crop],
                            f'diff_rasters/diff_crop_{crop}_t_{timestep_2}_{timestep_1}')

    return diff_rasters
