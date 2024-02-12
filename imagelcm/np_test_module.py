"""
Generates test outputs using a combination of numpy and vanilla Python 
to ensure that the LUE integration and allocation functions are
functioning properly.

NB: grazing intensities and management factors, as well as harvest
fractions are omitted in this test, as they are all constants and so do
not affect the behaviour of the functions as such.
"""

# import time
from math import ceil
import numpy as np
import parameters as prm

from read import check_wdir
import write as wt

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
    np.save(f"test_IO/suitmap_{nr}_{nc}_{shape}", suit_map)
    np.save(f"test_IO/is_land_{nr}_{nc}_{shape}", is_land)
    np.save(f"test_IO/is_cropland_{nr}_{nc}_{shape}", is_cropland)
    np.save(f"test_IO/regions_{nr}_{nc}_{shape}", regs)
    np.save(f"test_IO/g_area_{nr}_{nc}_{shape}", grid_area)
    np.save(f"test_IO/initial_fractions_{nr}_{nc}_{shape}", fractions)
    np.save(f"test_IO/demands_{nr}_{nc}_{shape}", demands)
    np.save(f"test_IO/potprod_{nr}_{nc}_{shape}", potential_prod)

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

    regs = np.load(f"test_IO/regions_{nr}_{nc}_{shape}.npy")
    fractions = np.load(f"test_IO/initial_fractions_{nr}_{nc}_{shape}.npy")
    demands = np.load(f"test_IO/demands_{nr}_{nc}_{shape}.npy")

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

    np.save(f"test_IO/fractions_first_reallocation_{nr}_{nc}_{shape}", new_fractions)

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

    np.save(f"test_IO/demand_maps_{nr}_{nc}_{shape}", demand_maps)

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

def denan_integration_input(array, dtype='float'):
    """
    Takes in inputs to integration functions and converts nans to zeros

    Parameters
    ----------
    array : np.ndarray
    dtype : str, default = 'float'
            datatype of array; must be either int or float

    Returns
    -------
    new_array : np.ndarray
                array of same dimensions as array
    """

    new_array = array.copy()
    if dtype=='float':
        new_array[np.isnan(new_array)] = 0.0
    elif dtype=='int':
        new_array[np.isnan(new_array)] = 0
    else:
        raise ValueError("dtype provided must be a string named 'int' or 'float'")

    return new_array

def integrate_r1_fractions(integrands, regs_flat, demand_maps, suit_map_flat, save=False, ir_yields=None):
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
    ir_yields : np.ndarray, default = None
                array of shape (NFC, N_REG) containing the total regional
                yields for all irrigated crops in all regions
    
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

    # convert nans to zeros
    integrands[np.isnan(integrands)] = 0.0

    # initialise regional prod array
    if nr<27:
        regional_prod = np.zeros((nr, nc+1))
    else:
        regional_prod = np.zeros((prm.N_REG, nc+1))

    # take into account irrigated crop yields, if applicable
    if ir_yields is not None:
        regional_prod[:, 1:] = ir_yields

    # loop through cells, from high suitability to low
    integrated_yields = np.zeros_like(integrands)
    for ind in sorted_inds:
        reg = regs_flat[ind]
        # if in valid region
        if reg!=np.nan and 0<reg<27:
            reg = int(reg)
            regional_prod[reg-1, :] += integrands[:, ind]
            integrated_yields[:, ind] = regional_prod[reg-1, :]

    # somehow turn 1D arrays back into 2D ones
    integrated_maps = np.zeros((nc+1, shape[0], shape[1]))
    for crop in range(nc+1):
        integrated_maps[crop, :, :] = np.reshape(integrated_yields[crop], shape)

    # compute Boolean demands_met maps
    demands_met = integrated_maps < demand_maps

    # still need to change values for yields > demands so it gives the same output as LUE function

    # save output arrays
    if save:
        np.save(f"test_IO/integrated_maps_{nr}_{nc}_{shape}", integrated_maps)
        np.save(f"test_IO/demands_met_{nr}_{nc}_{shape}", demands_met)
        np.save(f"test_IO/regional_prods_{nr}_{nc}_{shape}", regional_prod)

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

    np.save(f"test_IO/fracs_second_reallocation_{nr}_{nc}_{shape}", fracs_r2)

    return fracs_r2

def find_relevant_cells(is_cropland, expansion=False):
    """
    Determines cells that are to be (re)allocated

    PARAMETERS
    ----------
    is_cropland : np.ndarray
                  2D array of bools with value True for cells belonging to
                  the cropland LCT and False otherwise
    expansion : bool, default = False
                whether cropland is being expanded. If so, current
                cropland allocation will not be altered

    RETURNS
    -------
    cells_relevant : np.ndarray
                     flattened array stating whether cells are to be
                     (re)allocated
    """

    # map of cells that can be changed (if expansion, then non-cropland; if not, then cropland)
    cells_relevant = np.logical_xor(is_cropland.flatten(), expansion)

    return cells_relevant

def adjust_demand_surpassers(cell_fracs, prod_reg, yield_facs_cell, demands_reg, incl_grass=True):
    """
    Identify whether there are overproduced crops in a cell; fix as needed

    PARAMETERS
    ----------
    cell_fracs : np.ndarray
                 1D array of length # crops, containing the initial
                 allocations of crop fractions to either the  current cell
                 or part of the current cell remaining, following an
                 earlier re-working of the allocation
    prod_reg : np.ndarray
               1D array of length # crops, containing the crop production
               in the region to which the cell belongs, up to this cell
    yield_facs_cell : np.ndarray
                      1D array of length # crops, containing the factors
                      by which each crop fraction of the current cell
                      should be multiplied to find the projeced yield
    demands_reg : np.ndarray
                  1D array of length # crops, containing the crop demands
                  in the region to which the cell belongs
    incl_grass : bool, default = True
                 True if grass production is to be checked; False if not

    RETURNS
    -------
    cell_fracs : np.ndarray
                 updated crop fractions in the current cell
    """

    # find initial rdm for current region
    # rdm_reg = prod_reg >= demands_reg
    rdm_reg = (demands_reg - prod_reg) < prm.EPS

    # set overproduced crop fractions to 0
    cdm = np.where(rdm_reg)[0]
    cell_fracs[cdm] = 0.0

    # configure array slicing based on whether grass is included
    ind_1 = int(not incl_grass) # 0 if incl_grass, 1 if not

    ## Now considering cases where the current cell may make production exceed demand

    # store temporary updated regional production for current region
    reg_prod_temp = (prod_reg[ind_1:]
                        + cell_fracs[ind_1:] * yield_facs_cell[ind_1:])

    # store temporary regional demand met boolean array for current region and c_dm
    # rdm_temp = reg_prod_temp >= demands_reg[ind_1:]
    rdm_temp = (demands_reg[ind_1:] - reg_prod_temp) < prm.EPS
    c_dm = np.where(np.logical_and(rdm_temp, cell_fracs[ind_1:]>0))[0] + ind_1

    # correct fractions which have just made regional production exceed demand
    if len(c_dm):
        relevant_yfs = yield_facs_cell[c_dm]
        cell_fracs[c_dm] = ((demands_reg[c_dm] - prod_reg[c_dm])
                                        / relevant_yfs)
        # set divide by 0 to 0
        cell_fracs[np.where(relevant_yfs==0)[0]] = 0.0

    return cell_fracs

def integration_allocation(sdp_facs_flat, yield_facs_flat, old_fracs_flat, regs_flat, suit_map,
                           reg_demands, cells_relevant, initial_reg_prod=None, ir_frac=None):
    """
    Allocates new land (or remaining cropland fracs) based on summed yield

    Parameters
    ----------
    sdp_facs_flat : np.ndarray
                    array of shape (# crops, # cells in map) containing
                    sdpf=1000*grmppc/maxPR for each crop and cell, units
                    of [1000km^2 / T]
    yield_facs_flat : np.ndarray
                      array of shape (# crops, # cells in map) containing
                      the factor by which to multiply crop fractions to
                      compute the yield of the crop in that cell in [kT]
    old_fracs_flat : np.ndarray
                     array of shape (# crops, # cells in map) containing
                     the pre-allocated crop fractions to be checked and
                     altered accordingly
    regs_flat : np.ndarray
                flattened array stating the region to which each cell
                belongs
    suit_map : np.ndarray
               2D array containing the suitability of each cell in the map
    reg_demands : np.ndarray
                  array of shape (# regions, # crops) containing the crop
                  demands by region and crop
    cells_relevant : np.ndarray
                     flattened array stating whether cells are to be
                     (re)allocated
    initial_reg_prod : np.ndarray, default = None
                       array of shape (# regions, # crops) containing the
                       unchangeable cop production that is already
                       accounted for by region and crop. Unchangeable crop
                       production refers to irrigated crop yields if
                       expansion=False; crop yields from existing cropland
                       if expansion=True
    ir_frac : np.ndarray, default = None
              1D array containing the total fraction of each cell
              allocated to irrigated crops, used to ensure normalisation

    RETURNS
    -------
    new_fracs_flat : np.ndarray
                     array of shape (# crops, # latitudes * # longitudes)
                     containing the updated crop fractions
    reg_prod : np.ndarray
               array of shape (# regions, # crops) containing the updated
               regional production of each crop
    """
    if initial_reg_prod is None:
        reg_prod = np.zeros((prm.N_REG, prm.NGFC))
    else:
        reg_prod = initial_reg_prod.copy()

    if ir_frac is None:
        ir_frac = old_fracs_flat[0, :] * 0.0

    # indices, in order of declining suitability
    sorted_inds = find_sorted_indices(suit_map)
    print(f'# NaNs in sorted_indices: {np.count_nonzero(np.isnan(sorted_inds))}')

    # compute demands met boolean (one value for each region, for each crop)
    rdm = (reg_demands - reg_prod) < prm.EPS

    # loop through cells, from high suitability to low-
    extra_fracs = np.zeros_like(old_fracs_flat).astype(np.float32)
    yields = np.zeros_like(old_fracs_flat).astype(np.float32)
    for ind in sorted_inds:
        reg = regs_flat[ind]
        # if in valid region and cell belongs to the appropriate category (cropland or not)
        if reg!=np.nan:
            reg = int(reg)
            if 0<reg<=prm.N_REG and cells_relevant[ind]:
                # if demand not yet met in region
                if rdm[reg-1, :].size!=np.where(rdm[reg-1, :])[0].size:
                    # make sure demand not surpassed in current cell by old_fracs_flat
                    old_fracs_flat[:, ind] = adjust_demand_surpassers(old_fracs_flat[:, ind],
                                                                    reg_prod[reg-1, :],
                                                                    yield_facs_flat[:, ind],
                                                                    reg_demands[reg-1, :])

                    # compute potential yield, regional prod boolean array and sum of fractions
                    yields[:, ind] = old_fracs_flat[:, ind] * yield_facs_flat[:, ind]
                    reg_prod[reg-1, :] += yields[:, ind]
                    f_sum = old_fracs_flat[:, ind].sum()

                    # add irrigated fraction to the sum, to ensure normalisation
                    f_sum += ir_frac[ind]

                    # remaining regional demand for crop; 0 where demand met
                    dem_remain = reg_demands[reg-1, :] - reg_prod[reg-1, :]
                    dem_remain[np.where(rdm[reg-1, :])[0]] = 0.0

                    # compute 'sdempr'
                    sdp = (dem_remain * sdp_facs_flat[:, ind]).sum()

                    # allocate (rest of) new crop fractions (NOT GRASS!!!)
                    if sdp>prm.EPS:
                        extra_fracs[1:, ind] = ((1-f_sum) * sdp_facs_flat[1:, ind] / sdp
                                                * dem_remain[1:])

                    # de-NaN extra_fracs
                    extra_fracs[np.isnan(extra_fracs[:, ind]), ind] = 0.0

                    # make sure demand not surpassed in current cellby extra_fracs
                    extra_fracs[:, ind] = adjust_demand_surpassers(extra_fracs[:, ind],
                                                                reg_prod[reg-1, :],
                                                                yield_facs_flat[:, ind],
                                                                reg_demands[reg-1, :],
                                                                incl_grass=False)

                    # if there is remaining space in the cell, fill the remainder with grass
                    diff = (1-f_sum) - extra_fracs[:, ind].sum()
                    if diff>prm.EPS:
                        extra_fracs[0, ind] = diff

                    # compute and take into account extra fracs' contribution to production
                    extra_yields = extra_fracs[:, ind] * yield_facs_flat[:, ind]
                    yields[:, ind] += extra_yields
                    reg_prod[reg-1, :] += extra_yields

                    # recalculate demands met boolean
                    # rdm = reg_prod >= reg_demands
                    rdm = (reg_demands - reg_prod) < prm.EPS

        # break if demand met for all crops
        if rdm.size==np.where(rdm)[0].size:
            break

    print(f'unique extra fracs: {np.unique(extra_fracs)}')
    new_fracs_flat = old_fracs_flat + extra_fracs

    return new_fracs_flat, reg_prod

def fill_grass(fracs, ir_frac):
    """
    Fill remainder of cells featuring irrigated crops with grass

    PARAMETERS
    ----------
    fracs : np.ndarray
            2D array containing the crop fractions for grass and rain-fed
            food crops, with spatial coordinated reduced to one dimension
    ir_frac : np.ndarray
              1D array containing the total fraction of each cell
              allocated to irrigated crops, used to ensure normalisation
    
    RETURNS
    -------
    fracs : np.ndarray
            updated crop flattened crop fractions
    """

    # find the sum of all fractions in each cell
    f_sum = np.sum(fracs, axis=0) + ir_frac

    # identify cells with non-zero irrigated fractions where f_sum<1
    to_be_filled = np.where(np.logical_and(ir_frac>0, f_sum<1.0))[0]

    # compute and fill the remainder
    grass_fracs = fracs[0, :]
    grass_fracs[to_be_filled] += (1.0 - f_sum[to_be_filled])

    return fracs


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

def unflatten_rasters(flattened_raster_like, shape):
    """
    Unflattens raster-like arrays (or stacks of rasters) to 2D (3D) arrays

    Parameters
    ----------
    flatened_raster_like : np.ndarray
                           n=1- or 2-D array with the spatial dimensions
                           reduced to a single dimension, to be converted
                           to an array with dimensions n+1
    shape : tuple
            shape of the desired spatial dimenions in cells
    
    Returns
    -------
    raster_like : np.ndarray
                  array with 1 (spatial) dimension more than
                  flattened_raster_like
    """

    input_shape = flattened_raster_like.shape
    if len(input_shape)==1:
        raster_like = np.reshape(flattened_raster_like, shape)
    elif len(input_shape)==2:
        raster_like = np.stack([np.reshape(flattened_raster_like[ind, :], shape)
                                          for ind in range(input_shape[0])], axis=0)
    else:
        raise ValueError("Should only call unflatten_rasters on np.ndarray with 1 or 2 dimensions")

    return raster_like

def compute_largest_fraction_np(fracs, nr, c_bool, save=False, nan_map=None):
    """
    Finds crop with largest fraction (and what the fraction is), cellwise

    Parameters
    ----------
    fracs : np.ndarray
            3D array with shape (#n_crops+grass, # n_lat, #_lon)
    save : bool, default = False
           if true, save outputs in .npy format
    
    Returns
    -------
    mac : np.ndarray
          2D array containing the most allocated crop, for all cells
    mfrac : np.ndarray
            2D array containing the largest crop fraction, for all cells
    """

    nc = fracs.shape[0]-1
    shape = fracs.shape[1:]

    # turn nans to -1
    fracs[np.isnan(fracs)] = -1.

    mac = np.argmax(fracs, axis=0).astype(np.float32)
    mfrac = np.amax(fracs, axis=0)

    # turn results from all-nan slices to nan (most likely water)
    mac[mac==-1], mfrac[mfrac==-1] = np.nan, np.nan

    # make non-cropland have value -1
    mac[~c_bool], mfrac[~c_bool] = -1, -1.

    if nan_map is not None:
        mac[nan_map] = np.NaN

    if save:
        np.save(f"test_IO/mac_{nr}_{nc}_{shape}", mac)
        np.save(f"test_IO/mfrac_{nr}_{nc}_{shape}", mfrac)
        wt.write_np_raster('mac_1', mac)
        wt.write_np_raster('mfrac_1', mfrac)

    return mac, mfrac
