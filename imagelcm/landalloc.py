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
    for reg in range(1, 27):
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
        d_map_t = lfr.cast(greg, np.float32)
        print(f"loading demand for crop {crop}")
        for reg in range(prm.N_REG):
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
        dr_t = lfr.cast(greg, np.float32)
        print(f"loading demand for crop {crop}")
        for reg in range(prm.N_REG):
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

    g_intens_map = lfr.cast(greg, np.float32)
    for reg in range(prm.N_REG):
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
    mfs = mfs[:prm.NGFC, :]

    mf_maps = []
    for crop in range(prm.NGFC):
        mf_map = lfr.where(greg<27, lfr.cast(greg, np.float32), 0)
        print(f"loading management factors for crop {crop}")
        for reg in range(prm.N_REG):
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

    new_rasters['is_cropland'] = is_cropland

    # # isolate current cropland for every crop fraction
    # cc_frs = []
    # for crop in range(prm.NGFC):
    #     cc_frs.append(lfr.where(is_cropland, input_rasters['f'][crop]))

    # new_rasters['current_cropland'] = cc_frs

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
    has_irrigated : lue PartitionedArray<uint8>
                    raster of Booleans indicated where there is the
                    presence of irrigated cropland

    See Also
    --------
    isolate_cropland
    read.read_input_rasters
    """
    new_rasters = input_rasters

    fractions = input_rasters['f']
    has_irrigated = fractions[prm.NGFBC] > 0
    for crop in range(prm.NGFBC+1, prm.NGFBFC):
        has_irrigated = has_irrigated | (fractions[crop] > 0)

    return has_irrigated

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
    fh = nonraster_inputs['FH']

    # compute grazing intensity map
    graz_intens_map = return_grazintens_map(graz_intens, input_rasters["R"],
                                            input_rasters["R_bools"])

    # compute management factor and fharvcomb maps
    m_fac_maps = return_mf_maps(m_fac, input_rasters["R"], input_rasters["R_bools"])
    fh_maps = return_mf_maps(fh, input_rasters["R"], input_rasters["R_bools"])

    # write dictionary to return the maps
    mapped_data = {'demand_ratios':demand_ratios, 'current_dem_maps':current_demands,
                   'GI_map':graz_intens_map, 'MF_maps':m_fac_maps, 'FH_maps':fh_maps}

    return mapped_data

def compute_largest_fraction(fracs, is_cropland, save=False):
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

    wt.write_raster(broken_normalisation_bool, 'R1_normalisation_broken')

    # find crop with largest fraction for each cell
    _, ma_crop = compute_largest_fraction(new_fractions, input_rasters['is_cropland'])

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
    FH = mapped_data['FH_maps']
    MF_maps = mapped_data['MF_maps']
    GI_map = mapped_data['GI_map']

    pot_yields = [fracs[crop] * grmppc[crop] * max_pr[crop] * garea * FH[crop] * MF_maps[crop]
                 for crop in range(1, prm.NGFC)]
    pot_yields.insert(0, fracs[0] * grmppc[0] * max_pr[0] * garea * GI_map * FH[0] * MF_maps[0])

    return pot_yields

def compute_additional_rasters(input_rasters, mapped_data, nonraster_inputs):
    """
    Computes yield factor, sdp factor and cropland bool
    """

    grmppc = input_rasters['p_c']
    max_pr = nonraster_inputs['M']
    garea = input_rasters['A']
    FH = mapped_data['FH_maps']
    MF_maps = mapped_data['MF_maps']
    GI_map = mapped_data['GI_map']
    glct = input_rasters['lct']

    yield_facs = [grmppc[crop] * max_pr[crop] * garea * FH[crop] * MF_maps[crop]
                 for crop in range(1, prm.NGFC)]
    yield_facs.insert(0, grmppc[0] * max_pr[0] * garea * GI_map * MF_maps[0])

    sdp_facs = [1000 * grmppc[crop] / max_pr[crop] for crop in range(prm.NGFC)]

    is_cropland = glct == 1

    return yield_facs, sdp_facs, is_cropland

def raster_s_to_np(raster_s, is_list=False, flatten=False):
    """
    Converts a lue array or list of lue arrays to a numpy ndarray
    
    PARAMETERS
    ----------
    raster_s : lue PartitionedArray or list of lue PartitionedArray
    is_list : Bool, default = False
              True if raster_s is a list of lue arrays; false if it is a
              single array
    flatten : Bool
              True if the lue array(s) should be flattened

    RETURNS
    -------
    out : np.ndarray
          3-dimensional when is_list=True, flatten=false; 2-dimensional
          when is_list=False, flatten=False or when is_list=True,
          flatten=True; 1-dimensional when is_list=False, flatten=True
    """

    if is_list:
        if flatten:
            numpy_arrs = [lfr.to_numpy(arr).flatten for arr in raster_s]
        else:
            numpy_arrs = [lfr.to_numpy(arr) for arr in raster_s]
        out = np.asarray(numpy_arrs)
    else:
        out = lfr.to_numpy(raster_s)
        if flatten:
            out = out.flatten()

    return out

def prepare_for_np(to_be_converted):
    """
    Converts rasters to np format

    PARAMETERS
    ----------
    to_be_converted : dict
                      dictionary containing (lists of) lue
                      PartitionedArrays to be converted to np format
    
    RETURNS
    -------
    world_shape : tuple
                  shape of the IMAGE-LAND rasters
    converted : dict
                dictionary of arrays converted into np.ndarrays; some of
                them flattened
    """

    # dictionary to stor the converted arrays
    converted = {}

    # # convert (lists of) LUE arrays to numpy arrays; flatten
    # fracs_numpy = [lfr.to_numpy(arr) for arr in fracs]
    # fracs_np_3D = np.asarray(fracs_numpy)

    converted['fracs_3D'] = raster_s_to_np(to_be_converted['fracs'], is_list=True)


    # pyields_numpy = [lfr.to_numpy(arr) for arr in p_yields]
    # pyields_np_flat = np.asarray([py.flatten() for py in pyields_numpy])

    converted['flattened_yields'] = raster_s_to_np(to_be_converted['pot_yields'], is_list=True,
                                                   flatten=True)

    # dm_numpy = [lfr.to_numpy(arr) for arr in dem_maps.values()]
    # dm_2D = np.asarray(dm_numpy)
    # dm_np_flat = np.asarray([dn.flatten() for dn in dm_numpy])

    converted['dem_rats'] = raster_s_to_np(to_be_converted['dem_rats'], is_list=True)
    converted['flattened_dem_rats'] = raster_s_to_np(to_be_converted['dem_rats'], is_list=True,
                                                     flatten=True)

    # sm_np = lfr.to_numpy(suitmap) # don't flatten suitmap yet
    converted['suit'] = raster_s_to_np(to_be_converted['suit'])

    # regs_np_flat = lfr.to_numpy(regs).flatten()
    converted['flattened_R'] = raster_s_to_np(to_be_converted['suit'], flatten=True)

    # deal with nans in reg map
    converted['flattened_R'][np.isnan(converted['flattened_R'])] = 0

    world_shape = converted['suit'][0].shape

    return world_shape, converted

def compute_irrigated_yield(input_rasters, nonraster_inputs):
    """
    Computes total yield of irrigated cropland for each crop and region

    PARAMETERS
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    read_input_rasters
    non_raster_inputs : dict
                       dict containing the same items as that returned by
                       setup.

    RETURNS
    -------
    ir_yields : np.ndarray
                array of shape (NGFC, N_REG) containing the total regional
                yields for all irrigated crops in all regions
    """

    fracs = input_rasters['f']
    grmppc = input_rasters['p_c']
    garea = input_rasters['A']

    max_pr = nonraster_inputs['M']

    # multiply irrigated fraction raster by the area and corresponding grmppc rasters
    ir_raster_prods = [fracs[crop] * grmppc[crop] * garea for crop in range(prm.NGFBC, prm.NGFBFC)]

    # get regional totals of ir_raster_prods
    regional_tots = [lfr.zonal_sum(irrp, input_rasters['R']) for irrp in ir_raster_prods]

    ir_yields = np.zeros((prm.N_REG, prm.NGFC))
    for ind, rt in enumerate(regional_tots):
        # isolate totals for each region
        u, indices = np.unique(lfr.to_numpy(rt), return_index=True)
        corresponding_regions = lfr.to_numpy(input_rasters['R']).flatten()[indices]

        # make sure that regions 0 and 27 are removed
        where_valid_regions = np.logical_and(corresponding_regions>0,
                                             corresponding_regions<=prm.N_REG)
        u = u[where_valid_regions]
        corresponding_regions = corresponding_regions[where_valid_regions]

        ir_yields[corresponding_regions-1, ind+1] = u

    # compute true yields by multiplying ir_yields by M, MF and FH (reshaped for multiplication)
    ir_yields[:, 1:] *= (np.stack([max_pr[prm.NGFBC:] for _ in range(prm.N_REG)], 0)
                  * np.swapaxes((nonraster_inputs['MF'][1:prm.NGFC, :]
                                 * nonraster_inputs['FH'][prm.NFBC:, :]), 0, 1))

    np.save('outputs\\ir_yields', ir_yields)

    return ir_yields

def allocate_single_timestep(input_rasters, nonraster_inputs, ir_yields=None):
    """
    Runs the relevant functions to re-allocate all cropland once

    PARAMETERS
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    read_input_rasters
    non_raster_inputs : dict
                       dict containing the same items as that returned by
                       setup.
    ir_yields : np.ndarray
                array of shape (N_REG, NFGC) containing the regional crop
                yields of all irrigated crops
    """

    # map management facs, grazing intensity, demand ratios
    mapped_data = map_tabulated_data(input_rasters, nonraster_inputs)

    if prm.CHECK_IO:
        for key, value in mapped_data.items():
            if key=='GI_map':
                wt.write_raster(value, 'GI_map')
            else:
                if isinstance(value, dict):
                    for nkey, nvalue in value.items():
                        wt.write_raster(nvalue, f'{key}_{nkey}')
                else:
                    for ind, raster in enumerate(value):
                        wt.write_raster(raster, f'{key}_{ind}')

    # initial reallocation of existing cropland
    fracs_r1 = reallocate_cropland_initial(input_rasters, mapped_data)

    # compute potential yields
    pot_yields = compute_potential_yield(fracs_r1, input_rasters, mapped_data, nonraster_inputs)

    # (start LUE) computing (of) yield and sdp factors
    yield_facs, sdp_facs, is_cropland = compute_additional_rasters(input_rasters, mapped_data,
                                                                nonraster_inputs)

    # save first-stage crop reallocation, potential yield and is_cropland tiffs
    if prm.CHECK_IO:
        for crop, raster in enumerate(fracs_r1):
            wt.write_raster(raster, f'R1_crop_{crop}')
        for crop, raster in enumerate(pot_yields):
            wt.write_raster(raster, f'pot_yields_{crop}')
        wt.write_raster(is_cropland, f'is_cropland')

    # define dictionary of rasters to be converted to np.ndarray format
    rasters_for_numpy = {'fracs':fracs_r1, 'pot_yields':pot_yields,
                         'dem_rats':mapped_data['demand_ratios'], 'R':input_rasters['R'],
                         'suit':input_rasters['suit']}

    shp, np_rasters = prepare_for_np(rasters_for_numpy)

    print(f"map shape: {shp}")

    ############################## NUMPY SECTION ####################################
    # de-NaN
    integrands = ntm.denan_integration_input(np_rasters['flattened_yields'])

    # integrate
    integrated_yields, demands_met, regional_prod = ntm.integrate_r1_fractions(integrands,
                                                                               np_rasters['flattened_R'],
                                                                               np_rasters['dem_rats'],
                                                                               np_rasters['suit'])

    if prm.CHECK_IO:
        wt.write_np_raster('R1_integration_crop', integrated_yields)
        wt.write_np_raster('numpified_suitmap', np_rasters['suit'])
        for crop in range(np_rasters['flattened_yields'].shape[0]):
            wt.write_np_raster(f'R1_integrands_{crop}',
                               np.reshape(np_rasters['flattened_yields'][crop, :], shp))
        np.save('outputs\\regional_prods_R1', regional_prod)

    # get rid of fractions where demand has already been met
    fracs_r2 = ntm.rework_reallocation(np_rasters['fracs_3D'], demands_met,
                                       np_rasters['flattened_R'])

    # numpify/flatten inputs for integration-allocation - NEED TO SOMEHOW PUT THIS IN A FUNCTION
    yield_facs_flat = np.asarray([lfr.to_numpy(yf).flatten() for yf in yield_facs])
    sdpfs = np.asarray([lfr.to_numpy(sdpf).flatten() for sdpf in sdp_facs])
    fracs_r2_flat = ntm.flatten_rasters(fracs_r2)
    is_cropland = lfr.to_numpy(is_cropland)
    is_cropland_flat = is_cropland.flatten()

    # rearrange axes of tabulated demand_data
    reg_demands = np.swapaxes(nonraster_inputs['d'], 1, 2)

    # de-NaN
    sdpfs_denan = ntm.denan_integration_input(sdpfs)
    yfs_denan = ntm.denan_integration_input(yield_facs_flat)
    fracs_r2_denan = ntm.denan_integration_input(fracs_r2_flat)

    # get irrigation boolean raster
    irr_bool = lfr.to_numpy(get_irrigated_boolean(input_rasters))

    # call integration allocation on existing cropland. NB: irr yields are initial regional prod
    fracs_r3, reg_prod_updated = ntm.integration_allocation(sdpfs_denan, yfs_denan, fracs_r2_denan,
                                                            np_rasters['flattened_R'],
                                                            np_rasters['suit'], reg_demands,
                                                            np_rasters['dem_rats'],
                                                            is_cropland_flat, shp,
                                                            initial_reg_prod=ir_yields)

    # do land abandonment (cells w. 0 fractions for all crops & irrigation -> regr. forest (aband))

    # re-NaN
    should_be_nan = np.isnan(s_np)
    for crop in range(fracs_r3.shape[0]):
        fracs_r3[crop, should_be_nan] = np.nan
        wt.write_np_raster(f'new_fraction_{crop}', fracs_r3[crop, :, :])

    ntm.compute_largest_fraction_np(fracs_r3, prm.N_REG, is_cropland, save=True,
                                    nan_map=should_be_nan)

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

    # save initial crop fraction tifs
    if prm.CHECK_IO:
        for crop in range(len(input_rasters['f'])):
            wt.write_raster(input_rasters['f'][crop], f'initial_crop_{crop}')

        for ind, raster in enumerate(input_rasters['R_bools']):
            wt.write_raster(raster, f'reg_bool_{ind}')

        for ind, raster in enumerate(input_rasters['p_c']):
            wt.write_raster(raster, f'grmppc_{ind}')

        wt.write_raster(input_rasters['R'], 'regions_map')

    # save initial mac, mafrac
    compute_largest_fraction(input_rasters['f'], input_rasters['is_cropland'], save=True)

    # compute crop yields from irrigated land
    ir_yields = compute_irrigated_yield(input_rasters, nonraster_inputs)

    # allocate single timestep
    allocate_single_timestep(input_rasters, nonraster_inputs, ir_yields=ir_yields)

    print(f"Time taken: {time()-start}")

if __name__=="__main__":
    main()
