"""
Generates test outputs using a combination of numpy and vanilla Python 
to ensure that the LUE integration and allocation functions are
functioning properly.

NB: grazing intensities and management factors, as well as harvest
fractions are omitted in this test, as they are all constants and so do
not affect the behaviour of the functions as such.
"""

from math import ceil
import numpy as np
# import parameters as prm

from read import check_wdir

def write_inputs(shape=(10, 20), nr=1, nc=3):
    """
    Writes the input arrays for the test.
    
    Parameters
    ----------
    shape : tuple of ints, default = (10, 20)
            desired shape of the array, with orientation
            (longitude, latitude)
    nr : int
         number of regions (stand-ins for IMAGE world-regions)
    nc : int
         number of crops (not including grass)
    """

    suit_map = np.random.uniform(size=shape)
    is_land = np.random.uniform(size=shape) > 0.5
    is_cropland = np.logical_and(np.random.uniform(size=shape)>0.6, is_land)

    # make sure suitability of non-land (water) cells is 0
    suit_map[~is_land] = 0

    # find array's aspect ratio
    a_r = np.asarray(shape) / np.gcd(*shape)

    # find regions map
    regs = write_regs(shape, is_land, nr, a_r)

    # generate broadly reasonable area map
    grid = np.indices(shape)
    grid_area = np.sin((grid[0]+1) / (shape[0]+1) * np.pi) * 100

    # generate potential prod. such that it is highest at the equator
    potential_prod = np.stack([np.sin((grid[0]+1) / (shape[0]+1) * np.pi) for _ in range(nc+1)],
                              axis=0)
    potential_prod[~np.stack([is_land for _ in range(nc+1)], axis=0)] = 0.0

    # generate initial fractions
    fractions = generate_fractions(shape, is_cropland, nc)
    
    regional_prods = np.zeros((nr, nc+1))
    for reg in range(nr):
        for crop in range(nc+1):
            regional_prods[reg, crop] = (potential_prod[crop][regs==reg+1] * grid_area[regs==reg+1]
                                         * fractions[crop][regs==reg+1]).sum()

    demands = np.zeros((2, nr, nc+1))
    demands[0, :, :] = regional_prods
    demands[1, :, :] = np.random.normal(loc=1.1, scale=0.2, size=(nr, nc+1)) * demands[0, :, :]

    check_wdir()

    # save input files
    np.save(f"test_IO\\suitmap_{nr}_{nc}_{shape}", suit_map)
    np.save(f"test_IO\\is_land_{nr}_{nc}_{shape}", is_land)
    np.save(f"test_IO\\is_cropland_{nr}_{nc}_{shape}", is_cropland)
    np.save(f"test_IO\\regions_{nr}_{nc}_{shape}", regs)
    np.save(f"test_IO\\g_area_{nr}_{nc}_{shape}", grid_area)
    np.save(f"test_IO\\initial_fractions_{nr}_{nc}_{shape}", fractions)
    np.save(f"test_IO\\demands_{nr}_{nc}_{shape}", demands)
    np.save(f"test_IO\\potprod_{nr}_{nc}_{shape}", potential_prod)

def generate_fractions(shape, is_cropland, nc):
    """
    Generates initial fractions on initial cropland.

    Parameters
    ----------
    shape : tuple of ints, default = (20, 10)
            desired shape of the array
    is_cropland : np.ndarray
                  2D Boolean array stating whether each element is
                  cropland
    nc : int
         number of crops (not including grass)

    Returns
    -------
    fractions : np.ndarray
                3D array containing the crop fractions for every cell, for
                every crop
    """

    fractions = np.zeros((nc+1, shape[0], shape[1]))
    for crop in range(1, nc+1):
        fractions[crop] = np.random.uniform(high=1-np.max(np.sum(fractions[1:crop, :, :], axis=0)),
                                            size=shape)
        fractions[crop][~is_cropland] = 0
    fractions[0, :, :] = 1 - np.sum(fractions[1:, :, :], axis=0)
    fractions[0][~is_cropland] = 0

    return fractions

def write_regs(shape, is_land, nr, a_r):
    """
    Writes the regions input array for the test.
    
    Parameters
    ----------
    shape : tuple of ints, default = (20, 10)
            desired shape of the array
    is_land : np.ndarray
              2D Boolean array stating whether each element is land
    nr : int
         number of regions (stand-ins for IMAGE world-regions)
    a_r : np.ndarray
          np array of shape (1, 2) containing the aspect ratio of the array

    Returns
    -------
    regs : np.ndarray
           2D array of ints showing the region each array entry is in
    """

    # find number of 'rows' and 'columns' of regions
    scale_fac = ceil(np.sqrt(nr / a_r.prod()))
    regs_shape = a_r * scale_fac

    regs = np.ones(shape)
    i, j, xp1, yp1, reg = 0, 0, 0, 0, 0
    large_grid = shape[0]/regs_shape[0]
    while not (xp1==shape[0] and yp1==shape[1]):
        tmp_j = j
        j = reg % regs_shape[1]
        if j==0 and tmp_j!=0:
            i += 1

        x = round(large_grid * i)
        y = round(large_grid * j)
        xp1 = round(large_grid * (i + 1))
        yp1 = round(large_grid * (j + 1))
        if reg>=nr: # large region
            regs[x:xp1, y:yp1] *= nr
        else:
            regs[x:xp1, y:yp1] *= reg + 1

        # fill in next region
        reg += 1

    regs[~is_land] = 0

    return regs

def np_first_reallocation(shape=(10, 20), nr=1, nc=3):
    """
    Performs first step of reallocating existing cropland; saves result.
    
    Parameters
    ----------
    shape : tuple of ints, default = (10, 20)
            desired shape of the array, with orientation
            (longitude, latitude)
    nr : int, default = 1
         number of regions (stand-ins for IMAGE world-regions)
    nc : int, default = 3
         number of crops (not including grass)
    """

    check_wdir()

    regs = np.load(f"test_IO\\regions_{nr}_{nc}_{shape}.npy")
    fractions = np.load(f"test_IO\\initial_fractions_{nr}_{nc}_{shape}.npy")
    demands = np.load(f"test_IO\\demands_{nr}_{nc}_{shape}.npy")

    # calculate ratio of last demand to this timestep's demand
    demand_ratios = demands[1, :, :] / demands[0, :, :]

    # generate demand map
    demand_maps = return_npdemand_map(demand_ratios, regs, nr, nc)

    # calculate cf1 and cf2 and ensure zeros become nan values
    cf1 = np.sum(fractions * demand_maps, axis=0)
    cf3 = np.sum(fractions, axis=0)
    cf1[cf1==0] = np.nan
    cf3[cf3==0] = np.nan

    # compute new fractions from cf1 and cf3
    new_fractions = np.zeros_like(fractions)
    new_fractions[1:] = (fractions[1:] * demand_maps[1, :]
                         * np.stack([(cf3 / cf1) for _ in range(nc)], axis=0))

    # turn nan values back to zeros
    np.nan_to_num(new_fractions, copy=False, nan=0.0)

    # fill in the remainder with grass
    new_fractions[0] = 1 - np.sum(new_fractions[1:], axis=0)

    np.save(f"test_IO\\fractions_first_reallocation_{nr}_{nc}_{shape}", new_fractions)

def return_npdemand_map(demand_ratios, regs, nr=1, nc=3):
    """
    Puts demand ratios onto a 2D array, by region.
    
    Parameters
    ----------
    demand_ratios : np.ndarray
                    np array with shape (#regions, #crops+grass)
    regs : np.ndarray
           2D array of ints showing the region each array entry is in
    nr : int, default = 1
         number of regions (stand-ins for IMAGE world-regions)
    nc : int, default=3
         number of crops (not including grass)
    
    Returns
    -------
    demand_maps : np.ndarray
                  np array of shape (#crops+grass, #lat, #lon), containing
                  the ratio in demands for each cell (same value for all
                  cells in a given region)
    """

    shape = regs.shape
    demand_maps = np.zeros((nc+1, shape[0], shape[1]))
    for crop in range(nc+1):
        for reg in range(1, nr+1):
            demand_maps[crop][regs==reg] = demand_ratios[reg-1, crop]

    np.save(f"test_IO\\demand_maps_{nr}_{nc}_{shape}", demand_maps)

    return demand_maps

def find_sorted_indices(suit_map):
    """
    Finds indices to sort flattened arrays by suitability in desc. order

    Parameters
    ----------
    suit_map : np.ndarray
               2D np array containing the suitabilities for each cell

    Returns
    -------
    sorted_args : np.ndarray
                  1D np array with containing the indices for the
                  suitability-sorted arrays
    """

    suitmap_flat = suit_map.flatten()

    # indices, in order of declining suitability
    sorted_args = np.argsort(suitmap_flat)[::-1]

    return sorted_args

def integrate_r1_fractions(integrands, regs_flat, demand_maps, suit_map_flat, save=False):
    """
    Integrates fractions returned by the first reallocation of cropland

    Parameters
    ----------
    integrands : np.ndarray
                 np array of shape (n_crops+1, n_lat, n_lon) containing
                 the quantity to be integrated (crop fractions multiplied
                 by the potential yield, area and relevant constants)
    regs : np.ndarray
           np array containing the regions for every cell in the map
    demand_maps : np.ndarray
                  np array of shape (#crops+grass, #lat, #lon), containing
                  the ratio in demands for each cell (same value for all
                  cells in a given region)
    suit_map : np.ndarray
               2D np array containing the suitabilities for each cell
    
    Returns
    -------
    integrated_maps : np.ndarray
                      np array of shape (n_crops+1, n_lat, n_lon)
                      containing the integrated quantity (regional crop
                      production, summed over cells in order of
                      suitability)
    demands_met : np.ndarray
                  np array of booleans of same shape as fracs_r1 stating
                  whether the regional demand for each crop has been met
                  yet in each cell
    regional_prod : np.ndarray
                    np array of shape (n_regions, n_crops+1) containing
                    the production of each crop by each region
    """

    shape = demand_maps.shape[1:]
    nr = int(np.nanmax(regs_flat))
    # nc here defined as number of non-grass crops
    nc = integrands.shape[0]-1

    # indices, in order of declining suitability
    sorted_inds = find_sorted_indices(suit_map_flat)

    # loop through cells, from low suitability to high
    if nr<27:
        regional_prod = np.zeros((nr, nc+1))
    else:
        regional_prod = np.zeros((26, nc+1))
    integrated_yields = np.zeros_like(integrands)
    for ind in sorted_inds:
        reg = regs_flat[ind]
        # if in valid region
        if reg!=np.nan:
            if 0<reg<27:
                reg = int(reg)
                regional_prod[reg-1, :] += integrands[:, ind]
                integrated_yields[:, ind] = regional_prod[reg-1, :]

    # somehow turn 1D arrays back into 2D ones
    integrated_maps = np.zeros((nc+1, shape[0], shape[1]))
    for crop in range(nc+1):
        integrated_maps[crop] = np.reshape(integrated_yields[crop], shape)

    # compute Boolean demands_met maps
    demands_met = integrated_maps < demand_maps

    # still need to change values for yields > demands so it gives the same output as LUE function

    # save output arrays
    if save:
        np.save(f"test_IO\\integrated_maps_{nr}_{nc}_{shape}", integrated_maps)
        np.save(f"test_IO\\demands_met_{nr}_{nc}_{shape}", demands_met)
        np.save(f"test_IO\\regional_prods_{nr}_{nc}_{shape}", regional_prod)

    return integrated_maps, demands_met, regional_prod

def rework_reallocation(fracs_r1, demands_met, regs):
    """
    Delete reallocated fractions where demand is surpassed.

    Parameters
    ----------
    fracs_r1 : np.ndarray
               np array of shape (n_crops+1, n_lat, n_lon) containing
               the crop fractions resulting from the initial reallocation
               of existing cropland
    demands_met : np.ndarray
                  np array of booleans of same shape as fracs_r1 stating
                  whether the regional demand for each crop has been met
                  yet in each cell
    regs : np.ndarray
           np array containing the regions for every cell in the map

    Returns
    -------
    fracs_r2 : np.ndarray
               The resulting array once the fractions allocated to crops
               for which regional demand has already been met has been set
               to zero
    """

    shape = fracs_r1.shape[1:]
    nr = regs.max()
    nc = fracs_r1.shape[0]

    # fractions in the region in which demand is met should be set to 0
    fracs_r1[demands_met] = 0.0
    fracs_r2 = fracs_r1

    np.save(f"test_IO\\fracs_second_reallocation_{nr}_{nc}_{shape}", fracs_r2)

    return fracs_r2

def integration_allocation(sdp_facs_flat, yield_facs_flat, old_fracs_flat, regs_flat, suit_map,
                           reg_prod, reg_demands, dems_flat, is_cropland, shape, expansion=False):
    """
    Allocates new land (or remaining cropland fracs) based on summed yield

    Parameters
    ----------
    fracs_remaining : np.ndarray, default = None
                      2D array containing the remaining empty fractions
                      in each cell following the correction to the initial
                      cropland reallocation. Value defaults to None in the
                      case that the function is called to expand cropland

    """

    # map of cells that can be changed (if expansion, then non-cropland; if not, then cropland)
    cells_relevant = np.logical_xor(is_cropland, expansion)

    # shape = regs.shape
    nc = yield_facs_flat.shape[0]

    # indices, in order of declining suitability
    sorted_inds = find_sorted_indices(suit_map)

    # compute demands met boolean (one value for each region, for each crop)
    rdm = reg_prod > reg_demands[1, : , :]

    # loop through cells, from low suitability to high
    sdp = np.zeros_like(regs_flat)
    extra_fracs = np.zeros_like(old_fracs_flat)
    yields = np.zeros_like(old_fracs_flat)
    for ind in sorted_inds:
        reg = regs_flat[ind]
        # if in valid region and cell belongs to the appropriate category (cropland or not)
        if reg!=np.nan:
            reg = int(reg)
            if 0<reg<27 and cells_relevant[ind]:
                # find crops with demand already met - 'old' fraction must be 0
                c_dm = np.where(rdm[reg-1, :])[0]
                old_fracs_flat[c_dm, ind] = 0.0

                # store temporary updated regional production for current region
                reg_prod_temp = old_fracs_flat[:, ind] * yield_facs_flat[:, ind]
                # reg_prod[reg-1, :] += old_fracs_flat[:, ind] * yield_fac_flat[:, ind]

                # store temporary regional demand met boolean array for current region and c_dm
                rdm_temp = reg_prod_temp > reg_demands[1, reg-1 , :]
                c_dm = np.where(np.logical_and(rdm_temp, old_fracs_flat[:, ind]>0))[0]
                # print(c_dm)

                # correct fractions which have just made regional production exceed demand
                if len(c_dm):
                    old_fracs_flat[c_dm, ind] = ((reg_demands[1, reg-1, c_dm] - reg_prod[reg-1, c_dm])
                                                    / yield_facs_flat[c_dm, ind])
                    rdm[reg-1, c_dm] = True

                # compute potential yield, regional prod boolean array and sum of fractions
                yields[:, ind] = old_fracs_flat[:, ind] * yield_facs_flat[:, ind]
                reg_prod += yields[:, ind]
                f_sum = old_fracs_flat[:, ind].sum()
                # print(f_sum)

                # remaining regional demand for crop
                dem_remain = dems_flat[:, ind]-reg_prod[reg-1, :]
                sdp[ind] = (dem_remain * sdp_facs_flat[:, ind]).sum()
                extra_fracs[:, ind] = (old_fracs_flat[:, ind]
                                        + sdp_facs_flat[:, ind] / sdp[ind] * dem_remain)

    new_fracs_flat = old_fracs_flat + extra_fracs

    # reshape into raster-like shape
    new_fracs = np.zeros((nc, shape[0], shape[1]))
    for crop in range(nc):
        new_fracs[crop] = np.reshape(new_fracs_flat[crop], shape)

    # If some non-cropland has higher suitability than cropland, obvs don't want to expand
    # cropland when demand could potentially be met by existing cropland. Hence, need to call this
    # function twice - and the first time, to rejig existing computed fractions, will need the
    # is_cropland boolean as an input.

    return new_fracs, reg_prod

def flatten_rasters(raster_like):
    """
    Flattens raster-like arrays (or stacks of rasters) to 1D (2D) arrays

    Parameters
    ----------
    raster_like : np.ndarray
                      2D or 3D array to be converted into a 1D or 2D array
                      with the spatial dimensions being reduced to a
                      single dimension
    
    Returns
    -------
    flattened_raster_like : np.ndarray
                            array with 1 dimension fewer than raster_like
    """

    shape = raster_like.shape
    if len(shape)==2:
        flattened_raster_like = raster_like.flatten()
    elif len(shape)==3:
        flattened_raster_like = np.stack([raster_like[ind].flatten() for ind in range(shape[0])],
                                         axis=0)
    else:
        raise ValueError("Should only call flatten_rasters on np.ndarray with 2 or 3 dimensions.")

    return flattened_raster_like
