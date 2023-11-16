"""
Generates test outputs using a combination of numpy and vanilla Python 
to ensure that the LUE integration and allocation functions are
functioning properly.
"""

import numpy as np
from math import ceil
import parameters as prm

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

    # save input files
    np.save("test_IO\\suitmap", suit_map)
    np.save("test_IO\\is_land", is_land)
    np.save("test_IO\\is_cropland", is_cropland)
    np.save("test_IO\\regions", regs)
    np.save("test_IO\\g_area", grid_area)
    np.save("test_IO\\initial_fractions", fractions)
    np.save("test_IO\\demands", demands)

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



write_inputs(nr=2)
