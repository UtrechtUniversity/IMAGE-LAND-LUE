"""
Contains the functions to (re)allocate land for a single timestep.
"""
import numpy as np
import lue.framework as lfr
import lue.image_land as img

import parameters as prm
import read as rd
import write as wt
import np_test_module as ntm
import analysis as ans

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
        # ensure 0s in eg. Greenland
        d_map_t = lfr.where((d_map_t>prm.N_REG)|(d_map_t<1), 0., d_map_t)
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
    greg : lfr.PartitionedArray<int8>
           regions raster
    r_bools : list
              list containing the boolean rasters computed by setup, in
              lue format
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

    d_t = demands[timestep, :, :]
    d_tm1 = demands[timestep-1, :, :]

    # NaN prevention: as in original IMAGE-LAND, 0 previous demand => dem ratio is set to 0
    d_t[d_tm1==0] = 0.
    d_tm1[d_tm1==0] = 1.

    ratio_dict = {}

    for crop in range(prm.NGFC):
        dr_t = lfr.cast(greg, np.float32)
        print(f"loading demand for crop {crop}")
        # ensure 0s in eg. Greenland
        dr_t = lfr.where((dr_t>prm.N_REG)|(dr_t<1), 0., dr_t)
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
    greg : lfr.PartitionedArray<int8>
           regions raster
    r_bools : list
              list containing the boolean rasters computed by setup, in
              lue format
    
    Returns
    -------
    g_intens_map : lue data object
                   lue array containing a map of grazing intensities

    See Also
    --------
    return_demand_rasters
    setup
    """

    # assume all grazing intensive, for now
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
    greg : lfr.PartitionedArray<int8>
           regions raster
    r_bools : list
              list containing the boolean maps computed by setup, in
              lue format
    
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
    Finds extent of existing cropland using land cover type raster.

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    
    Returns
    -------
    new_rasters : dict
                 dict containing the same items as input_rasters, with 
                 an added Boolean raster showing the extent of existing
                 cropland, and a list of the rain-fed food and fodder crop
                 fractions 

    See Also
    --------
    setup               
    """

    # find cropland boolean raster
    is_cropland = input_rasters['lct'] == 1

    # # isolate current cropland for every crop fraction
    # cc_frs = []
    # for crop in range(prm.NGFC):
    #     cc_frs.append(lfr.where(is_cropland, input_rasters['f'][crop]))

    # new_rasters['current_cropland'] = cc_frs

    return is_cropland

def map_tabulated_data(input_rasters, nonraster_inputs, timestep=1):
    """
    Maps non-spatial data (eg. management facs) onto rasters in lue format

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    nonraster_inputs : Dict
                       dict containing the same items as that returned by
                       setup
    timestep : int, default = 1
               current timestep

    RETURNS
    -------
    mapped_data : dict
                  contains demand_ratios, a dictionary of lue arrays with
                  the ratio in crop demand during the current and previous
                  timestep for every crop, mapped onto corresponding world
                  regions; demands, equivalent to demand_ratios but with
                  the current demands rather than the ratio in demand;
                  grazing intensities mapped onto corresponding regions;
                  management factors mapped onto corresponding regions;
                  frharvcombs mapped onto corresponding regions

    See Also
    --------
    setup
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

def reallocate_cropland_initial(input_rasters, mapped_data, ir_frac):
    """
    Executes the first step of reallocating existing cropland.

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    mapped_data : dict
                  the output of map_tabulated_data
    ir_frac : lfr.PartitionedArray<float32>
              raster-like array containing the total fraction of each cell
              allocated to irrigated crops, used to ensure normalisation
    
    Returns
    -------
    new_fractions : list of lue data objects
                    list containing rasters for each new crop fraction.
                    NB: only for grass and rain-fed fod crops

    See Also
    --------
    setup
    map_tabulated_data
    """

    # relabel relevant input variables
    fractions = input_rasters['f']

    # compute cf_1 and cf_3
    cf1 = fractions[0] * mapped_data['demand_ratios']['crop0'] / mapped_data['GI_map']
    cf3 = fractions[0]
    for crop in range(1, prm.NGFC):
        cf1 += (fractions[crop]
                * mapped_data['demand_ratios'][f'crop{crop}'] / mapped_data['MF_maps'][crop])
        cf3 += fractions[crop]

    # compute new fractions from cf_1 and cf_3, as well as the sum of the new fractions
    new_fractions = []
    cf_ratio = cf3 / cf1
    for crop in range(1, prm.NGFC):
        new_fraction = (fractions[crop]  * mapped_data['demand_ratios'][f'crop{crop}']
                        / mapped_data['MF_maps'][crop] * cf_ratio)
        new_fractions.append(new_fraction)

        # update sum of fractions
        if crop==1:
            fraction_sum = new_fraction
        else:
            fraction_sum += new_fraction

    # add sum of irrigated fractions to fraction_sum
    fraction_sum += ir_frac
    unique_fsum = np.unique(lfr.to_numpy(fraction_sum))
    print(f'max frac after initial allocation: {unique_fsum}')

    # ensure the sum of no cell fraction exceeds 1
    broken_normalisation_bool = fraction_sum > 1

    # if prm.CHECK_IO:
    wt.write_raster(broken_normalisation_bool, 'R1_normalisation_broken')

    # find crop with largest fraction for each cell
    _, ma_crop = ans.compute_largest_fraction(new_fractions, input_rasters['is_cropland'])

    # remove excess fraction from largest cell fraction; set fraction_sum to 1 in those cells
    for crop in range(0, prm.NFC):
        # cells in which fraction_sum>1 and 'crop' is the crop with the largest fraction
        to_be_changed = broken_normalisation_bool & (ma_crop == crop)
        new_fractions[crop] = lfr.where(to_be_changed, new_fractions[crop]-(fraction_sum-1),
                                        new_fractions[crop])
        fraction_sum = lfr.where(to_be_changed, 1.0, fraction_sum)

        unique_fractions = np.unique(lfr.to_numpy(new_fractions[crop]))
        print(f'unique corrected fractions after initial realloc: {unique_fractions}')

    unique_fsum = np.unique(lfr.to_numpy(fraction_sum))
    print(f'max frac after normalisation correction: {unique_fsum}')

    # rest of the cells are allocated to grass
    new_fractions.insert(0, 1-fraction_sum)

    return new_fractions

def compute_projected_yield(fracs, input_rasters, mapped_data, nonraster_inputs):
    """
    Compute projected yield raster-like LUE arrays in [kT]

    Parameters
    ----------
    fracs : list
            list of lue arrays containing the crop fractions following
            their recalculation in reallocate_cropland_initial
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    mapped_data : dict
                  the output of map_tabulated_data
    nonraster_inputs : dict
                       dict containing the same items as that returned by
                       setup

    Returns
    -------
    proj_yields : list
                 list of lue arrays, 1 per crop, containing the projected
                 crop yields in every cell, based on current fractions

    See Also
    --------
    setup
    map_tabulated_data
    reallocated_cropland_initial
    """

    grmppc = input_rasters['p_c']
    max_pr = nonraster_inputs['M']
    garea = input_rasters['A']
    f_harv = mapped_data['FH_maps']
    mf_maps = mapped_data['MF_maps']
    gi_map = mapped_data['GI_map']

    # crop-1 for f_harv as it has no grass value
    proj_yields = [fracs[crop] * grmppc[crop] * max_pr[crop] * garea * f_harv[crop-1]
                  * mf_maps[crop] / 1000 for crop in range(1, prm.NGFC)]

    # grass
    if prm.MF_AS_GI:
        proj_yields.insert(0, fracs[0] * grmppc[0] * max_pr[0] * garea * mf_maps[0] / 1000)
    else:
        proj_yields.insert(0, fracs[0] * grmppc[0] * max_pr[0] * garea * gi_map / 1000)

    return proj_yields

def compute_additional_rasters(input_rasters, mapped_data, nonraster_inputs):
    """
    Computes yield factor, sdp factor and cropland bool

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    mapped_data : dict
                  the output of map_tabulated_data
    nonraster_inputs : dict
                       dict containing the same items as that returned by
                       setup

    Returns
    -------
    yield_facs : list
                 list of lue PartitionedArrays containing the factor by
                 which to multiply crop fractions to compute the yield
                 of the crop in that cell in [kT] ([Gg])
    sdp_facs : list
               list of lue PartitionedArrays containing the factor by
               which to multiply remaining crop demands, which can in turn
               be summed to compute the quantity known as sdempr.
               [sdp_facs] = [1000km^2 / T]

    See Also
    --------
    setup
    map_tabulated_data
    """

    grmppc = input_rasters['p_c']
    max_pr = nonraster_inputs['M']
    garea = input_rasters['A']
    f_harv = mapped_data['FH_maps']
    mf_maps = mapped_data['MF_maps']
    gi_map = mapped_data['GI_map']

    # crop-1 for f_harv as it has no grass value
    yield_facs = [grmppc[crop] * max_pr[crop] * garea * f_harv[crop-1] * mf_maps[crop] / 1000
                 for crop in range(1, prm.NGFC)]

    # grass
    if prm.MF_AS_GI:
        yield_facs.insert(0, grmppc[0] * max_pr[0] * garea * mf_maps[0] / 1000)
    else:
        yield_facs.insert(0, grmppc[0] * max_pr[0] * garea * gi_map / 1000)

    sdp_facs = [1000 * grmppc[crop] / max_pr[crop] for crop in range(prm.NGFC)]

    return yield_facs, sdp_facs

def raster_s_to_np(raster_s, is_list=False, flatten=False):
    """
    Converts a lue array or list of lue arrays to a numpy ndarray
    
    Parameters
    ----------
    raster_s : lue PartitionedArray or list of lue PartitionedArray
    is_list : Bool, default = False
              True if raster_s is a list of lue arrays; false if it is a
              single array
    flatten : Bool
              True if the lue array(s) should be flattened

    Returns
    -------
    out : np.ndarray
          3-dimensional when is_list=True, flatten=false; 2-dimensional
          when is_list=False, flatten=False or when is_list=True,
          flatten=True; 1-dimensional when is_list=False, flatten=True
    """

    if is_list:
        if flatten:
            numpy_arrs = [lfr.to_numpy(arr).flatten() for arr in raster_s]
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

    Parameters
    ----------
    to_be_converted : dict
                      dictionary containing (lists of) lue
                      PartitionedArrays to be converted to np format
    
    Returns
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

    converted['yields_flat'] = raster_s_to_np(to_be_converted['pot_yields'], is_list=True,
                                                   flatten=True)

    # dm_numpy = [lfr.to_numpy(arr) for arr in dem_maps.values()]
    # dm_2D = np.asarray(dm_numpy)
    # dm_np_flat = np.asarray([dn.flatten() for dn in dm_numpy])

    converted['dem_rats'] = raster_s_to_np(to_be_converted['dem_rats'].values(), is_list=True)
    converted['dem_rats_flat'] = raster_s_to_np(to_be_converted['dem_rats'].values(), is_list=True,
                                                     flatten=True)

    # UNIQUE = np.unique(converted['dem_rats_flat'])

    # sm_np = lfr.to_numpy(suitmap) # don't flatten suitmap yet
    converted['suit'] = raster_s_to_np(to_be_converted['suit'])

    # regs_np_flat = lfr.to_numpy(regs).flatten()
    converted['R_flat'] = raster_s_to_np(to_be_converted['R'], flatten=True)

    # deal with nans in reg map
    converted['R_flat'][np.isnan(converted['R_flat'])] = 0

    converted['yf_flat'] = raster_s_to_np(to_be_converted['yf'], is_list=True, flatten=True)

    converted['sdpf_flat'] = raster_s_to_np(to_be_converted['sdpf'], is_list=True,
                                                 flatten=True)

    converted['is_cropland'] = raster_s_to_np(to_be_converted['is_cropland'])
    converted['is_cropland_flat'] = raster_s_to_np(to_be_converted['is_cropland'], flatten=True)

    converted['IR_frac'] = raster_s_to_np(to_be_converted['IR_frac'], flatten=True)

    world_shape = converted['suit'].shape

    return world_shape, converted

def rework_reallocation_lue(fracs_r1, integrated_yields, demand_rasters):
    """
    Delete reallocated fractions where demand is surpassed.

    Parameters
    ----------
    fracs_r1 : list
               list of length prm.NGFC lue arrays containing the crop
               fractions resulting from the initial reallocation of
               existing cropland
    integrated_yields : list
                        list of length prm.NGFC lue arrays containing the
                        integrated yield raster for each crop returned by
                        lue.image_land.integrate()
    demand_rasters : list
                     list of length prm.NGFC lue arrays containing the
                     demands for each crop type, mapped onto each
                     corresponding region

    Returns
    -------
    fracs_r2 : list
               List of the resulting arrays once the fractions allocated 
               to crops for which regional demand has already been met has
               been set to zero
    """

    fracs_r2 = [lfr.where(integrated_yields[crop_idx]>demand_rasters[crop_idx],
                          0.0,
                          fracs_r1[crop_idx])
                for crop_idx in range(len(fracs_r1))]

    return fracs_r2

def back_to_lue(arr, dtype='float'):
    """
    Converts np arrays (mostly new crop fractions) back to the LUE format

    Parameters
    ----------
    arr : np.ndarray
          np array with 2 or 3 dimensions
    dtype : str, default = 'float'
            datatype of the elements of arr
    
    Returns
    -------
    lue_out : lue PartitionedArray<> or list
              if arr is 2D, returns PartitionedArray; if arr is 3D,
              returns a list of PartionedArrays
    """
    if dtype=='float':
        arr = arr.astype(np.float32)

    shp = arr.shape
    if len(shp)==3:
        lue_out = [lfr.from_numpy(arr[ind, :, :], partition_shape=prm.PART_SHP)
                   for ind in range(shp[0])]
    elif len(shp)==2:
        lue_out = lfr.from_numpy(arr, partition_shape=prm.PART_SHP)
    else:
        raise ValueError("back_to_lue() input must be a 2- or 3-dimensional array ")

    return lue_out

def update_lcts(fracs, glct):
    """
    Changes LCTs to match re-allocated land [conversion is intantaneous]

    Parameters
    ----------
    fracs : list
            list of lue PartitionedArrays containing the most recent crop
            fractions
    glct : lue PartitionedArray<>
           array containing the land cover types for every grid cell
    
    Returns
    -------
    is_cropland : lue PartitionedArray<>
                  new array of booleans showing the extent of updated
                  cropland
    new_glct : lue PartitionedArray<>
               array containing the updated land cover types for every
               grid cell
    """

    print(glct)

    # compute updated cropland boolean
    # for crop in range(prm.NGFBFC):
    for crop in range(prm.NGFC):
        if (0<crop<prm.NGFC) or crop>prm.NFBC: # if not biofuel
            if crop==1:
                is_cropland = fracs[crop] > 0
            else:
                is_cropland = is_cropland | (fracs[crop] > 0)

    print(is_cropland)

    # ensure that cropland really is cropland
    new_glct = lfr.where(is_cropland, 1., glct) # this is where something goes wrong

    # ensure that newly abandoned cropland becomes regrowth forest (abandoning)
    not_cropland = ~is_cropland
    perceived_cropland = glct==1
    newly_abandoned_cropland = not_cropland & perceived_cropland
    new_glct = lfr.where(newly_abandoned_cropland, 4., new_glct)

    return is_cropland, new_glct

def allocate_single_timestep_part_lue(input_rasters, nonraster_inputs, timestep, ir_info=None):
    """
    Runs the relevant functions to re-allocate all land for one timestep

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    non_raster_inputs : dict
                       dict containing the same items as that returned by
                       setup.
    timestep : int
               current model timestep
    ir_info : dict, default = None
              dictionary containing the irrigated crop yields, the areas
              in which there is irrigated cropland and the irrigated crop
              fractions (isolated from the rain-fed crop fractions)

    Returns
    -------
    new_rasters : dict
                  contains a list of reallocated crop fractions across the
                  whole map and a lue array with the reallocated land
                  cover types (LCTs) in every grid cell
    reg_prod_updated : np.ndarray
                       crop production per region

    See Also
    --------
    setup
    reallocate_cropland_initial
    """
    # unpack ir_info
    ir_yields = ir_info['ir_yields']
    ir_bool = ir_info['ir_bool']
    ir_frac = ir_info['ir_frac']

    # isolate cropland
    input_rasters['is_cropland'] = isolate_cropland(input_rasters)

    # map management facs, grazing intensity, demand ratios
    mapped_data = map_tabulated_data(input_rasters, nonraster_inputs, timestep)

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
    fracs_r1 = reallocate_cropland_initial(input_rasters, mapped_data, ir_frac)

    # combine reallocated fracs with biofuel and irrigated crop fractions (python list joining)
    all_fracs_r1 = fracs_r1 + input_rasters['f'][prm.NGFC:]

    # compute projected yields
    proj_yields = compute_projected_yield(fracs_r1, input_rasters, mapped_data, nonraster_inputs)

    # (start LUE) computing (of) yield and sdp factors
    yield_facs, sdp_facs = compute_additional_rasters(input_rasters, mapped_data, nonraster_inputs)

    # save first-stage crop reallocation, projected yield and is_cropland tiffs
    if prm.CHECK_IO:
        for crop, raster in enumerate(fracs_r1):
            wt.write_raster(raster, f'R1_crop_{crop}')
        for crop, raster in enumerate(proj_yields):
            wt.write_raster(raster, f'pot_yields_{crop}')
        wt.write_raster(input_rasters['is_cropland'], 'is_cropland')

    # define dictionary of rasters to be converted to np.ndarray format
    rasters_for_numpy = {'fracs':all_fracs_r1, 'pot_yields':proj_yields,
                         'dem_rats':mapped_data['demand_ratios'], 'R':input_rasters['R'],
                         'suit':input_rasters['suit'], 'yf':yield_facs, 'sdpf':sdp_facs,
                         'is_cropland':input_rasters['is_cropland'], 'IR_frac':ir_frac}

    shp, np_rasters = prepare_for_np(rasters_for_numpy)

    print(f"map shape: {shp}")

    ############################## NUMPY SECTION ####################################
    # de-NaN
    np_rasters['yields_flat'][np.isnan(np_rasters['yields_flat'])] = 0.0
    print(np_rasters['yields_flat'].dtype)
    integrands = ntm.denan_integration_input(np_rasters['yields_flat'])

    # integrate
    integrated_yields, dems_met, regional_prod = ntm.integrate_r1_fractions(integrands,
                                                                            np_rasters['R_flat'],
                                                                            np_rasters['dem_rats'],
                                                                            np_rasters['suit'])

    if prm.CHECK_IO:
        wt.write_np_raster('R1_integration_crop', integrated_yields)
        wt.write_np_raster('numpified_suitmap', np_rasters['suit'])
        for crop in range(np_rasters['yields_flat'].shape[0]):
            wt.write_np_raster(f'R1_integrands_{crop}',
                               np.reshape(np_rasters['yields_flat'][crop, :], shp))
        np.save('outputs/regional_prods_R1', regional_prod)

    # get rid of fractions where demand has already been met
    fracs_r2 = ntm.rework_reallocation(np_rasters['fracs_3D'][:prm.NGFC, :, :], dems_met,
                                       np_rasters['R_flat'])

    # flatten inputs for integration-allocation
    fracs_r2_flat = ntm.flatten_rasters(fracs_r2)

    # isolate appropriate timestep and rearrange axes of tabulated demand_data
    reg_demands = np.swapaxes(nonraster_inputs['d'][timestep, :, :], 0, 1)

    # de-NaN
    sdpfs_denan = ntm.denan_integration_input(np_rasters['sdpf_flat'])
    yfs_denan = ntm.denan_integration_input(np_rasters['yf_flat'])
    fracs_r2_denan = ntm.denan_integration_input(fracs_r2_flat)

    if prm.CHECK_IO:
        for crop in range(sdpfs_denan.shape[0]):
            wt.write_np_raster(f'sdpf_{crop}', np.reshape(np_rasters['sdpf_flat'][crop, :], shp))
            wt.write_np_raster(f'yf_{crop}', np.reshape(np_rasters['yf_flat'][crop, :], shp))

    # get irrigation boolean raster
    ir_bool = lfr.to_numpy(ir_bool)

    # determine cells to be altered by integration_allocation
    cells_relevant = ntm.find_relevant_cells(np_rasters['is_cropland'])

    # call integration allocation on existing cropland. NB: irr yields are initial regional prod
    fracs_r3, reg_prod_updated = ntm.integration_allocation(sdpfs_denan, yfs_denan, fracs_r2_denan,
                                                            np_rasters['R_flat'],
                                                            np_rasters['suit'], reg_demands,
                                                            cells_relevant,
                                                            initial_reg_prod=ir_yields,
                                                            ir_frac=np_rasters['IR_frac'])

    # determine whether expansion is necessary
    demands_met = reg_prod_updated >= (reg_demands - prm.EPS)
    expand = np.count_nonzero(demands_met) < demands_met.size

    if expand:
        # determine cells to be altered by integration_allocation
        cells_relevant = ntm.find_relevant_cells(np_rasters['is_cropland'], expansion=True)

        # call integration-allocation on non-cropland
        fracs_r3, reg_prod_updated = ntm.integration_allocation(sdpfs_denan, yfs_denan,
                                                                fracs_r3, np_rasters['R_flat'],
                                                                np_rasters['suit'], reg_demands,
                                                                cells_relevant,
                                                                initial_reg_prod=reg_prod_updated)

    # reshape new crop fraction array
    fracs_r3 = ntm.unflatten_rasters(fracs_r3, shp)

    # concatenate new fractions array with irrigated fractions
    fracs_r3 = np.concatenate((fracs_r3, np_rasters['fracs_3D'][prm.NGFC:, :, :]), axis=0)

    # saving new fracs
    wt.write_np_raster(f"frac_rasters/new_fraction_{shp}_t{timestep}", fracs_r3)

    # re-NaN
    should_be_nan = np.isnan(np_rasters['suit'])
    for crop in range(fracs_r3.shape[0]):
        fracs_r3[crop, should_be_nan] = np.nan
        wt.write_np_raster(f'new_fraction_{crop}', fracs_r3[crop, :, :])
    print(np.unique(fracs_r3))

    # convert back to LUE
    new_fracs = back_to_lue(fracs_r3)

    # do land abandonment (cells w. 0 fractions for all crops & irrigation -> regr. forest (aband))
    is_cropland, new_glct = update_lcts(new_fracs, input_rasters['lct'])

    # save current mac, mafrac
    ans.compute_largest_fraction(new_fracs, is_cropland, save=True, timestep=timestep)

    # return as dictionary
    new_rasters = {'fracs':new_fracs, 'lct':new_glct}

    return new_rasters, reg_prod_updated

def allocate_single_timestep_full_lue(input_rasters, nonraster_inputs, timestep, ir_info=None):
    """
    Runs the relevant functions to re-allocate all land for one timestep

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    non_raster_inputs : dict
                       dict containing the same items as that returned by
                       setup.
    timestep : int
               current model timestep
    ir_info : dict, default = None
              dictionary containing the irrigated crop yields, the areas
              in which there is irrigated cropland and the irrigated crop
              fractions (isolated from the rain-fed crop fractions)

    Returns
    -------
    new_rasters : dict
                  contains a list of reallocated crop fractions across the
                  whole map and a lue array with the reallocated land
                  cover types (LCTs) in every grid cell
    reg_prod_updated : np.ndarray
                       crop production per region

    See Also
    --------
    setup
    reallocate_cropland_initial
    """
    # unpack ir_info
    ir_yields = ir_info['ir_yields']
    ir_bool = ir_info['ir_bool']
    ir_frac = ir_info['ir_frac']

    # isolate cropland
    input_rasters['is_cropland'] = isolate_cropland(input_rasters)

    # map management facs, grazing intensity, demand ratios
    mapped_data = map_tabulated_data(input_rasters, nonraster_inputs, timestep)

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
    fracs_r1 = reallocate_cropland_initial(input_rasters, mapped_data, ir_frac)

    # compute projected yields
    proj_yields = compute_projected_yield(fracs_r1, input_rasters, mapped_data, nonraster_inputs)

    # (start LUE) computing (of) yield and sdp factors
    yield_facs, sdp_facs = compute_additional_rasters(input_rasters, mapped_data, nonraster_inputs)

    # save first-stage crop reallocation, projected yield and is_cropland tiffs
    if prm.CHECK_IO:
        for crop, raster in enumerate(fracs_r1):
            wt.write_raster(raster, f'R1_crop_{crop}')
        for crop, raster in enumerate(proj_yields):
            wt.write_raster(raster, f'pot_yields_{crop}')
        wt.write_raster(input_rasters['is_cropland'], 'is_cropland')

    # compute max cells
    max_cells = int(prm.SHP[0]*prm.SHP[1]* 0.7 / 5)

    ###### compute suitability-ordered routes #####
    # route over all land
    route = lfr.decreasing_order(input_rasters['R'], input_rasters['suit'], max_cells)

    # route over cropland only
    suit_map_cropland_only = lfr.where(input_rasters['is_cropland'], input_rasters['suit'])
    regions_cropland_only = lfr.where(input_rasters['is_cropland'], input_rasters['R'])

    route_cropland_only = lfr.decreasing_order(regions_cropland_only,
                                               suit_map_cropland_only,
                                               max_cells)

    # route over non-cropland only
    suit_map_non_cropland_only = lfr.where(~input_rasters['is_cropland'], input_rasters['suit'])
    regions_non_cropland_only = lfr.where(~input_rasters['is_cropland'], input_rasters['R'])

    route_non_cropland_only = lfr.decreasing_order(regions_non_cropland_only,
                                                   suit_map_non_cropland_only,
                                                   max_cells)

    # first integration
    # integrated_yields = []
    # for proj_yield in proj_yields:
    #     integrated_yields.append(img.integrate(route, proj_yield, max_cells))

    # get rid of fractions where demand has already been met
    # fracs_r2 = rework_reallocation_lue(fracs_r1, integrated_yields, mapped_data{'current_dem_maps'})

    # isolate appropriate timestep and rearrange axes of tabulated demand_data
    reg_demands = np.swapaxes(nonraster_inputs['d'][timestep, :, :], 0, 1)

    # convert ir_yields and reg_demands to lue format
    zones = [reg for reg in range(1, prm.N_REG+1)]
    ir_yields_lue = lfr.make_ready_future(
                    {
                        zone: [ir_yields[zone-1, crop_idx] for crop_idx in range(prm.NGFC)]
                        for zone in zones
                    }
                )
    reg_demands_lue = lfr.make_ready_future(
                    {
                        zone: [reg_demands[zone-1, crop_idx] for crop_idx in range(prm.NGFC)]
                        for zone in zones
                    }
                )

    ##### adjust reallocation / abandon not needed cropland #####
    (fracs_r3, reg_prod_updated) = img.integrate_and_allocate(route, sdp_facs, yield_facs,
                                                              fracs_r1, reg_demands_lue,
                                                              ir_yields_lue, ir_frac)

    # make program wait for reg_prod_updated to be calculated, a value and no longer a future
    reg_prod_updated = reg_prod_updated.get()

    # convert dictionary containing lists into 2D np array, via Python list of lists
    reg_prod_updated = np.asarray(list(reg_prod_updated.values()))

    # determine whether expansion is necessary
    demands_met = reg_prod_updated >= (reg_demands - prm.EPS)
    expand = np.count_nonzero(demands_met) < demands_met.size

    if expand:
        # # determine cells to be altered by integration_allocation
        # cells_relevant = ntm.find_relevant_cells(np_rasters['is_cropland'], expansion=True)

        # # call integration-allocation on non-cropland
        # fracs_r3, reg_prod_updated = ntm.integration_allocation(sdpfs_denan, yfs_denan,
        #                                                         fracs_r3, np_rasters['R_flat'],
        #                                                         np_rasters['suit'], reg_demands,
        #                                                         cells_relevant,
        #                                                         initial_reg_prod=reg_prod_updated)

        # here need to do integrate_and_allocate following non-cropland route, using
        # reg_prod_updated where ir_yields_lue was used in the last integrate_and_allocate call.
        pass

    # concatenate new fractions array with irrigated fractions (python list joining)
    new_fracs = fracs_r3 + input_rasters['f'][prm.NGFC:]

    # saving new fracs
    for crop_idx in range(len(new_fracs[:prm.NGFC])):
        wt.write_raster(f"frac_rasters/new_fraction_{crop_idx}t{timestep}", new_fracs[crop_idx])

    # check unique fraction values to make sure they're between 0 and 1
    # print(np.unique(fracs_r3))

    # do land abandonment (cells w. 0 fractions for all crops & irrigation -> regr. forest (aband))
    is_cropland, new_glct = update_lcts(new_fracs, input_rasters['lct'])

    # save current mac, mafrac
    ans.compute_largest_fraction(new_fracs, is_cropland, save=True, timestep=timestep)

    # return as dictionary
    new_rasters = {'fracs':new_fracs, 'lct':new_glct}

    return new_rasters, reg_prod_updated
