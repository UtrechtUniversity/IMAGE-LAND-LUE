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
    potential_prod = np.sin((grid[0]+1) / (shape[0]+1) * np.pi)
    potential_prod[~is_land] = 0.0

    # generate initial fractions
    fractions = generate_fractions(shape, is_cropland, nc)
    
    regional_prods = np.zeros((nr, nc+1))
    for reg in range(nr):
        for crop in range(nc+1):
            regional_prods[reg, crop] = (potential_prod[regs==reg+1] * grid_area[regs==reg+1]
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

def integrate_r1_fractions(shape=(10, 20), nr=1, nc=3):
    """
    Integrates fractions returned by the first reallocation of cropland

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

    fracs_r1 = np.load(f"test_IO\\fractions_first_reallocation_{nr}_{nc}_{shape}.npy")
    regs = np.load(f"test_IO\\regions_{nr}_{nc}_{shape}.npy")
    suit_map = np.load(f"test_IO\\suitmap_{nr}_{nc}_{shape}.npy")
    demand_maps = np.load(f"test_IO\\demand_maps_{nr}_{nc}_{shape}.npy")

    # flatten arrays so they can be sorted
    fracs_flat = np.zeros((nc+1, shape[0]*shape[1]))
    for crop in range(nc+1):
        fracs_flat[crop, :] = fracs_r1[crop, :, :].flatten()
    regs_flat = regs.flatten()

    # indices, in order of declining suitability
    sorted_inds = find_sorted_indices(suit_map)

    # loop through cells, from low suitability to high
    regional_prod = np.zeros((nr, nc+1))
    integrated_yields = np.zeros_like(fracs_flat)
    for ind in sorted_inds:
        reg = int(regs_flat[ind])
        # if in valid region
        if reg>0:
            regional_prod[reg-1, :] += fracs_flat[:, ind]
            integrated_yields[:, ind] = regional_prod[reg-1, :]

    # somehow turn 1D arrays back into 2D ones
    integrated_maps = np.zeros((nc+1, shape[0], shape[1]))
    for crop in range(nc+1):
        integrated_maps[crop] = np.reshape(integrated_yields[crop], shape)

    # compute Boolean demands_met maps
    demands_met = integrated_maps < demand_maps

    # still need to change values for yields > demands so it gives the same output as LUE function

    np.save(f"test_IO\\integrated_maps_{nr}_{nc}_{shape}", integrated_maps)
    np.save(f"test_IO\\demands_met_{nr}_{nc}_{shape}", demands_met)
    np.save(f"test_IO\\regional_prods_{nr}_{nc}_{shape}", regional_prod)

def rework_reallocation(shape=(10, 20), nr=1, nc=3):
    """
    Delete reallocated fractions where demand is surpassed.

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

    fracs_r1 = np.load(f"test_IO\\fractions_first_reallocation_{nr}_{nc}_{shape}.npy")
    demands_met = np.load(f"test_IO\\demands_met_{nr}_{nc}_{shape}.npy")
    is_cropland = np.load(f"test_IO\\is_cropland_{nr}_{nc}_{shape}.npy")

    # fractions in the region in which demand is met should be set to 0
    fracs_r1[demands_met] = 0.0

    # find remaining unallocated land within existing cropland
    fracs_remain = 1 - np.sum(fracs_r1, axis=0)
    fracs_remain[~is_cropland] = 0.0

    np.save(f"test_IO\\fracs_second_reallocation_{nr}_{nc}_{shape}", fracs_r1)
    np.save(f"test_IO\\fracs_remaining_{nr}_{nc}_{shape}", fracs_remain)

def compute_sdp(shape=(10, 20), nr=1, nc=3):
    """
    Calculates SDP quantity (sdempr in OG code)

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

    pot_prod = np.load(f"test_IO\\potprod_{nr}_{nc}_{shape}.npy")
    fracs_r1 = np.load(f"test_IO\\fractions_first_reallocation_{nr}_{nc}_{shape}.npy")
    reg_prod = np.load(f"test_IO\\regional_prods_{nr}_{nc}_{shape}.npy")
    regs = np.load(f"test_IO\\regions_{nr}_{nc}_{shape}.npy")
    suit_map = np.load(f"test_IO\\suitmap_{nr}_{nc}_{shape}.npy")
    # integrated_maps = np.save(f"test_IO\\integrated_maps_{nr}_{nc}_{shape}.npy")
    demand_maps = np.load(f"test_IO\\demand_maps_{nr}_{nc}_{shape}.npy")

    # indices, in order of declining suitability
    sorted_inds = find_sorted_indices(suit_map)

    # flatten some arrays
    regs_flat = regs.flatten()
    fracs_flat = np.stack([fracs_r1[crop].flatten for crop in range(nc+1)], axis=0)
    dems_flat = np.stack([demand_maps[crop].flatten for crop in range(nc+1)], axis=0)
    pot_prods_flat = np.stack([pot_prod[crop].flatten for crop in range(nc+1)], axis=0)

    # find ind where cropland ends
    start_ind = 100

    # loop through cells, from low suitability to high
    sdp = np.zeros_like(regs_flat)
    for ind in sorted_inds[start_ind:]:
        reg = int(regs_flat[ind])
        # if in valid region
        if reg>0:
            reg_prod[reg-1, :] += fracs_flat[:, ind]
            sdp[ind] = ((dems_flat[:, ind]-reg_prod[reg-1, :]) * pot_prods_flat[:, ind]).sum()

def integration_allocation(fracs_remaining, shape=(10, 20), nr=1, nc=3):
    pass


write_inputs(nr=2)
np_first_reallocation(nr=2)
integrate_r1_fractions(nr=2)
rework_reallocation(nr=2)
# compute_sdp(nr=2)
