"""
Calls the functions in read.py and allocates land.
"""

from time import time

import numpy as np
import lue.framework as lfr

import parameters as prm
import read as rd
import write as wt
import np_test_module as ntm

def setup():
    """
    Calls the read functions and computes regional Boolean rasters.

    Returns
    -------
    input_rasters : Dict
                    dict containing the same items as that returned by
                    read.read_input_rasters
    nonraster_inputs : Dict
                       dict containing the same items as that returned by
                       read.read_nonraster_inputs, with the addition of a
                       list of Boolean rasters with values corresponding
                       to each IMAGE World Region

    See Also
    --------
    read.read_input_rasters
    read_nonraster_inputs
    """

    input_rasters = rd.read_input_rasters()
    nonraster_inputs = rd.read_nonraster_inputs()

    # compute regional boolean maps
    greg = input_rasters["R"]
    reg_bools = []
    for reg in range(26):
        reg_bools.append(greg == reg)
    input_rasters["R_bools"] = reg_bools

    return input_rasters, nonraster_inputs

def return_demand_rasters(demands, greg, r_bools, timestep):
    """
    Puts the food and fodder demands onto a raster, by World Region.
    
    Parameters
    ----------
    demands : np.ndarray
              np array with shape (# timesteps, # G+F crops, # regions)
    greg : lue data object
           regions raster
    r_bools : list of lue data objects
              list containing the boolean maps computed by setup
    timestep : int
               the current timestep
    
    Returns
    -------
    demand_dict_t : dict
                    dict containing a map of regional crop demands for
                    every food crop and grass

    See Also
    --------
    setup
    """

    d_t = demands[timestep, :, :]

    demand_dict_t = {}

    for crop in range(17):
        d_map_t = greg
        print(f"loading demand for crop {crop}")
        for reg in range(26):
            local_d_t = d_t[crop, reg]
            # if in correct region, value in every cell is equal to the demand in that region
            d_map_t = lfr.where(r_bools[reg], local_d_t, d_map_t)

        demand_dict_t[f"crop{crop}"] = d_map_t

    return demand_dict_t

def return_demand_ratios(demands, greg, r_bools, timestep):
    """
    Puts the increase in food and fodder demand onto a raster, by region.
    
    Parameters
    ----------
    demands : np.ndarray
              np array with shape (# timesteps, # G+F crops, # regions)
    greg : lue data object
           regions raster
    r_bools : list of lue data objects
              list containing the boolean maps computed by setup
    timestep : int
               the current timestep
    
    Returns
    -------
    demand_dict_t : dict
                    dict containing a map of the fractional increase in
                    regional crop demands for every food crop and grass,
                    relative to the previous timestep

    See Also
    --------
    return_demand_rasters
    setup
    """

    d_t = demands[:, timestep, :]
    d_tm1 = demands[:, timestep-1, :]

    ratio_dict = {}

    for crop in range(prm.NGFC):
        dr_t = greg
        print(f"loading demand for crop {crop}")
        for reg in range(26):
            local_ratio = d_t[crop, reg] / d_tm1[crop, reg]
            # if in correct region, value in every cell is equal to the demand in that region
            dr_t = lfr.where(r_bools[reg], local_ratio, dr_t)

        ratio_dict[f"crop{crop}"] = dr_t

    return ratio_dict

def return_grazintens_map(g_intens, greg, r_bools):
    """
    Puts the grazing intensities onto a raster, by region. UNFINISHED
    
    Parameters
    ----------
    g_intens : np.ndarray
               np array with shape (# grazing systems, # regions)
    greg : lue data object
           regions raster
    r_bools : list of lue data objects
              list containing the boolean maps computed by setup
    
    Returns
    -------
    g_intens_map : lue data object
                   lue array containing a map of grazing intensities

    See Also
    --------
    return_demand_rasters
    setup
    """

    # assume all grazing regime 1, for now
    g_intens = g_intens[0, :]

    g_intens_map = greg
    for reg in range(26):
        # if in correct region, value in each cell is equal to the grazing intensity in that region
        g_intens_map = greg = lfr.where(r_bools[reg], g_intens[reg], g_intens_map)

    return g_intens_map

def return_mf_maps(mfs, greg, r_bools):
    """
    Puts the management factors onto rasters, by region. UNFINISHED
    
    Parameters
    ----------
    mfs : np.ndarray
          np array with shape (# crops, # regions)
    greg : lue data object
           regions raster
    r_bools : list of lue data objects
              list containing the boolean maps computed by setup
    
    Returns
    -------
    g_intens_map : list of lue data object
                   list of lue arrays containing a map of management
                   factors for each food crop

    See Also
    --------
    return_demand_rasters
    setup
    """

    # assume all crops are rain-fed, for now
    mfs = mfs[:prm.NFC, :]

    mf_maps = []
    for crop in range(prm.NFC):
        mf_map = greg
        print(f"loading management factors for crop {crop}")
        for reg in range(26):
            # if in correct region, value in every cell is equal to management fac in that region
            mf_map = lfr.where(r_bools[reg], mfs[crop, reg], mf_map)
        mf_maps.append(mf_map)

    return mf_maps

def isolate_cropland(input_rasters):
    """
    Isolates existing cropland using land cover type raster.

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    read_input_rasters.
    
    Returns
    -------
    new_rasters : dict
                 dict containing the same items as input_rasters, with 
                 an added Boolean raster showing the extent of existing
                 cropland, and a list of the rain-fed food and fodder crop
                 fractions 

    See Also
    --------
    read.read_input_rasters                 
    """
    new_rasters = input_rasters

    # find cropland boolean raster
    is_cropland = input_rasters['lct'] == 1

    # isolate current cropland for every crop fraction
    cc_frs = []
    for crop in range(prm.NGFC):
        cc_frs.append(lfr.where(is_cropland, input_rasters['f'][crop]))

    new_rasters['current_cropland'] = cc_frs

    return new_rasters

def get_irrigated_boolean(input_rasters):
    """
    Identifies cells in which there is irrigated cropland.
    
    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    read.read_input_rasters.

    Returns
    -------
    new_rasters : dict
                  dict containing the same items as input_rasters, with 
                  an added Boolean raster showing the extent of irrigated
                  cropland.

    See Also
    --------
    isolate_cropland
    read.read_input_rasters
    """
    new_rasters = input_rasters

    fractions = input_rasters['f']
    has_irrigated = fractions[prm.NGFBC] > 0
    for crop in range(prm.NGFBC+1, prm.NGFBFC):
        has_irrigated = has_irrigated & fractions[crop] > 0

    new_rasters['IRB'] = has_irrigated

    return new_rasters

def allocated_irrigated_land():
    """Allocates a fraction of rain-fed cropland to irrigated land."""

def map_tabulated_data(input_rasters, nonraster_inputs, timestep=1):
    """
    Takes in tabular datasets (eg. management facs) and puts them on maps
    """

    # compute ratio of current demand to previous demand
    demand_ratios = return_demand_ratios(nonraster_inputs['d'], input_rasters["R"],
                                         input_rasters["R_bools"], timestep)
    
    current_demands = return_demand_rasters(nonraster_inputs['d'], input_rasters['R'],
                                           input_rasters['R_bools'], timestep)

    # relabel relevant input variables
    graz_intens = nonraster_inputs['GI']
    m_fac = nonraster_inputs['MF']

    # compute grazing intensity map
    graz_intens_map = return_grazintens_map(graz_intens, input_rasters["R"],
                                            input_rasters["R_bools"])
    
    # change to management maps!! (likely same problem for m_fac)
    m_fac_maps = return_mf_maps(m_fac, input_rasters["R"], input_rasters["R_bools"])

    # write dictionary to return the maps
    mapped_data = {'demand_ratios':demand_ratios, 'current_dem_maps':current_demands,
                   'GI_map':graz_intens_map, 'MF_maps':m_fac_maps}

    return mapped_data

def compute_largest_fraction(fracs, save=False):
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

    n_fracs = len(fracs)

    # set initial values to 1st crop('s fractions)
    ma_frac = fracs[0]
    ma_crop = lfr.where(fracs[0]>0, 1, 0)

    # update by iterating through all crops
    for crop in range(1, n_fracs):
        current_crop_bigger = fracs[crop] > ma_frac
        ma_frac = lfr.where(current_crop_bigger, fracs[crop], ma_frac)
        ma_crop = lfr.where(current_crop_bigger, crop, ma_crop)

    if save:
        wt.write_raster(ma_frac, 'mafrac')
        wt.write_raster(ma_crop, 'mac')

    return ma_frac, ma_crop

def reallocate_cropland_initial(input_rasters, mapped_data):
    """
    Executes the first step of reallocating existing cropland.

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    read_input_rasters, with the addition of the crop
                    demand increase rasters output by return_demand_ratios
                    and the existing cropland rasters output by
                    isolate_cropland.
    non_raster_inputs : dict
                       dict containing the same items as that returned by
                       setup.
    mapped_data : dict
    
    Returns
    -------
    new_fractions : list of lue data objects
                    list containing rasters for each new crop fraction
    """

    # relabel relevant input variables
    fractions = input_rasters['f']

    # compute cf_1 and cf_3
    cf1 = fractions[0] * mapped_data['demand_ratios']['crop0'] / mapped_data['GI_map']
    cf3 = fractions[0]
    for crop in range(1, prm.NGFC):
        cf1 += fractions[crop] * mapped_data['demand_ratios'][f'crop{crop}'] / mapped_data['MF_maps'][crop-1]
        cf3 += fractions[crop]

    # compute new fractions from cf_1 and cf_3, as well as the sum of the new fractions
    new_fractions = []
    cf_ratio = cf3 / cf1
    for crop in range(1, prm.NGFC):
        new_fraction = (fractions[crop] * mapped_data['MF_maps'][crop-1]
                        * mapped_data['demand_ratios'][f'crop{crop}'] * cf_ratio)
        new_fractions.append(new_fraction)

        # update sum of fractions
        if crop==1:
            fraction_sum = new_fraction
        else:
            fraction_sum += new_fraction

    # ensure the sum of no cell fraction exceeds 1
    broken_normalisation_bool = fraction_sum > 1

    # find crop with largest fraction for each cell
    _, ma_crop = compute_largest_fraction(new_fractions)

    # # determine whether normalisation has been broken anywhere - only do the next bit if so
    # norm_broken_anywhere = lfr.global_sum(lfr.where(broken_normalisation_bool, 1, 0)) > 0
    # if norm_broken_anywhere:
    # Would have to stop lue operations for above lines to work.

    # remove excess fraction from largest cell fraction; set fraction_sum to 1 in those cells
    for crop in range(1, prm.NFC):
        # cells in which fraction_sum>1 and 'crop' is the crop with the largest fraction
        to_be_changed = broken_normalisation_bool & (ma_crop == crop)
        new_fractions[crop] = lfr.where(to_be_changed, new_fractions[crop]-fraction_sum+1,
                                        new_fractions[crop])
        fraction_sum = lfr.where(to_be_changed, 1.0, fraction_sum)

    # rest of the cells are allocated to grass
    new_fractions.insert(0, 1-fraction_sum)

    return new_fractions

def compute_potential_yield(fracs, input_rasters, mapped_data, nonraster_inputs):
    """
    Compute potential yield raster-like LUE arrays
    """

    grmppc = input_rasters['p_c']
    max_pr = nonraster_inputs['M']
    garea = input_rasters['A']
    FH = nonraster_inputs['FH']
    MF_maps = mapped_data['MF_maps']
    GI_map = mapped_data['GI_map']

    pot_yields = [fracs[crop+1] * grmppc[crop+1] * max_pr[crop+1] * garea * FH[crop] * MF_maps[crop]
                 for crop in range(prm.NFC)]
    pot_yields.insert(0, fracs[0] * grmppc[0] * max_pr[0] * garea * GI_map)

    return pot_yields

def compute_additional_maps(input_rasters, mapped_data, nonraster_inputs):
    """
    Computes yield factor, sdp factor and cropland bool
    """

    grmppc = input_rasters['p_c']
    max_pr = nonraster_inputs['M']
    garea = input_rasters['A']
    FH = nonraster_inputs['FH']
    MF_maps = mapped_data['MF_maps']
    GI_map = mapped_data['GI_map']
    glct = input_rasters['lct']

    yield_facs = [grmppc[crop+1] * max_pr[crop+1] * garea * FH[crop] * MF_maps[crop]
                 for crop in range(prm.NFC)]
    yield_facs.insert(0, grmppc[0] * max_pr[0] * garea * GI_map)

    sdp_facs = [1000 * grmppc[crop] / max_pr[crop] for crop in range(prm.NGFC)]

    is_cropland = glct == 1

    return yield_facs, sdp_facs, is_cropland

def prepare_for_np(fracs, p_yields, dem_maps, regs, suitmap):
    """
    Converts rasters to np format.
    """

    # convert (lists of) LUE arrays to numpy arrays; flatten
    fracs_numpy = [lfr.to_numpy(arr) for arr in fracs]
    fracs_np_3D = np.asarray(fracs_numpy)
    fracs_np_flat = np.asarray([fn.flatten() for fn in fracs_numpy])

    pyields_numpy = [lfr.to_numpy(arr) for arr in p_yields]
    pyields_np_flat = np.asarray([py.flatten() for py in pyields_numpy])

    dm_numpy = [lfr.to_numpy(arr) for arr in dem_maps.values()]
    dm_2D = np.asarray(dm_numpy)
    dm_np_flat = np.asarray([dn.flatten() for dn in dm_numpy])

    sm_np_flat = lfr.to_numpy(suitmap).flatten()
    regs_np_flat = lfr.to_numpy(regs).flatten()

    # deal with nans in reg map
    regs_np_flat[np.isnan(regs_np_flat)] = 0

    world_shape = fracs_numpy[0].shape

    return (world_shape, fracs_np_flat, fracs_np_3D, pyields_np_flat, dm_np_flat, dm_2D,
            sm_np_flat, regs_np_flat)

@lfr.runtime_scope
def main():
    """main function"""

    rd.check_wdir()

    if not prm.STANDALONE:
        rd.prepare_input_files()
    rd.prepare_input_files()

    start = time()
    input_rasters, nonraster_inputs = setup()
    input_rasters = isolate_cropland(input_rasters)

    # save initial mac, mafrac
    compute_largest_fraction(input_rasters['f'])

    # map management facs, grazing intensity, demand ratios
    mapped_data = map_tabulated_data(input_rasters, nonraster_inputs)

    # initial reallocation of existing cropland
    fracs_r1 = reallocate_cropland_initial(input_rasters, mapped_data)

    # compute potential yields
    pot_yields = compute_potential_yield(fracs_r1, input_rasters, mapped_data, nonraster_inputs)

    # (start LUE) computing (of) yield and sdp factors
    yield_facs, sdp_facs, is_cropland = compute_additional_maps(input_rasters, mapped_data,
                                                                nonraster_inputs)

    shp, f_np, f_3D, py_np, d_np, dm_2D, s_np, r_np = prepare_for_np(fracs_r1, pot_yields,
                                                                     mapped_data['demand_ratios'],
                                                                     input_rasters['R'],
                                                                     input_rasters['suit'])

    print(f"map shape: {shp}")

    ############################## NUMPY SECTION ####################################
    # integrate
    _, demands_met, regional_prod = ntm.integrate_r1_fractions(py_np, r_np, dm_2D, s_np)

    # get rid of fractions where demand has already been met
    fracs_r2 = ntm.rework_reallocation(f_3D, demands_met, r_np)

    # numpify/flatten inputs for integration-allocation
    yield_facs_flat = np.asarray([lfr.to_numpy(yf).flatten() for yf in yield_facs])
    sdpfs = np.asarray([lfr.to_numpy(sdpf).flatten() for sdpf in sdp_facs])
    fracs_r2_flat = ntm.flatten_rasters(fracs_r2)
    is_cropland_flat = lfr.to_numpy(is_cropland).flatten()

    # rearrange axes of tabulated demand_data
    reg_demands = np.swapaxes(nonraster_inputs['d'], 1, 2)

    # call integration allocation on existing cropland
    fracs_r3, reg_prod_updated = ntm.integration_allocation(sdpfs, yield_facs_flat, fracs_r2_flat,
                                                            r_np, s_np, regional_prod,
                                                            reg_demands, d_np, is_cropland_flat,
                                                            shp, save=True)

    ntm.compute_largest_fraction_np(fracs_r3, 26, save=True)

    ## Might be something off with the number of crops!

    print(f"Time taken: {time()-start}")

if __name__=="__main__":
    main()
