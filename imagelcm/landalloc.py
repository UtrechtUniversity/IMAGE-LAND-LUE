"""
Central file of IMAGE-LAND-LUE. Run this file to run the model.
"""

from time import time

import numpy as np
import lue.framework as lfr

import parameters as prm
import read as rd
import write as wt
import np_test_module as ntm
import analysis as ans
import single_timestep as st

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

    # initial values of mf, fharvcomb and fraz intensity
    nonraster_inputs['MF'] = nonraster_inputs['MF_all'][0, :, :]
    nonraster_inputs['GI'] = nonraster_inputs['GI_all'][0, :, :]
    nonraster_inputs['FH'] = nonraster_inputs['FH_all'][0, :, :]

    # compute regional boolean maps
    greg = input_rasters["R"]
    reg_bools = []
    for reg in range(1, 27):
        reg_bools.append(greg == reg)
    input_rasters["R_bools"] = reg_bools

    return input_rasters, nonraster_inputs

def get_irrigated_boolean(input_rasters):
    """
    Identifies cells in which there is irrigated cropland.
    
    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup

    Returns
    -------
    has_irrigated : lue PartitionedArray<uint8>
                    raster of Booleans indicated where there is the
                    presence of irrigated cropland

    See Also
    --------
    isolate_cropland
    setup
    """

    fractions = input_rasters['f']
    has_irrigated = fractions[prm.NGFBC] > 0
    for crop in range(prm.NGFBC+1, prm.NGFBFC):
        has_irrigated = has_irrigated | (fractions[crop] > 0)

    return has_irrigated

def compute_ir_frac(input_rasters):
    """
    Computes total fraction allocated to irrigated crops in each cell
    
    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup

    Returns
    -------
    ir_frac : lue PartitionedArray<float32>
              raster containing total irrigated crop fractions

    See Also
    --------
    get_irrigated_boolean
    setup
    """

    # separate out irrigated fractions
    ir_fractions = input_rasters['f'][prm.NGFBC:]
    n_crops = prm.NGFBFC - prm.NGFBC

    # sum
    ir_frac = ir_fractions[0]
    for crop in range(1, n_crops):
        ir_frac += ir_fractions[crop]

    return ir_frac

def compute_irrigated_yield(input_rasters, nonraster_inputs):
    """
    Computes total yield of irrigated cropland for each crop and region

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    non_raster_inputs : dict
                       dict containing the same items as that returned by
                       setup.

    Returns
    -------
    ir_yields : np.ndarray
                array of shape (NGFC, N_REG) containing the total regional
                yields for all irrigated crops in all regions, in [kT]

    See Also
    --------
    setup
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
    ir_yields[:, 1:] *= (np.stack([max_pr[prm.NGFBC:] for _ in range(prm.N_REG)], 0) / 1000
                         * np.swapaxes((nonraster_inputs['MF'][1:prm.NGFC, :]
                                        * nonraster_inputs['FH'][prm.NFBC:, :]), 0, 1))

    np.save('outputs/ir_yields', ir_yields)

    return ir_yields

def perform_allocation_loop(input_rasters, nonraster_inputs, ir_info, n_step, invl=1):
    """
    (Re)allocates land for number of timesteps n_step at an interval invl

    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    setup
    non_raster_inputs : dict
                       dict containing the same items as that returned by
                       setup
    ir_info : dict
              dictionary containing the irrigated crop yields, the areas
              in which there is irrigated cropland and the irrigated crop
              fractions (isolated from the rain-fed crop fractions)
    n_step : int
             the number of timesteps over which the allocation should be
             performed
    invl : int, default = 1
           the desired interval between each timestep [years]. For
           example, n_step=3 and invl=5 will lead to the model being run
           over years 5, 10 and 15.

    See Also
    --------
    setup
    allocate_single_timestep
    """
    # compute and save initial crop areas
    crop_areas =  ans.compute_crop_areas(input_rasters['f'], input_rasters['A'],
                                    input_rasters['R'])
    np.save('outputs/crop_areas_0', crop_areas)

    for timestep in range(invl, 1+invl*n_step, invl):
        # update mf, fharvcomb and graz intensity
        nonraster_inputs['MF'] = nonraster_inputs['MF_all'][timestep-1, :, :]
        nonraster_inputs['GI'] = nonraster_inputs['GI_all'][timestep-1, :, :]
        nonraster_inputs['FH'] = nonraster_inputs['FH_all'][timestep-1, :, :]

        # allocate single timestep
        if prm.FULL_LUE:
            new_rasters, reg_prod = st.allocate_single_timestep_full_lue(input_rasters,
                                                                         nonraster_inputs,
                                                                         timestep, ir_info)
        else:
            new_rasters, reg_prod = st.allocate_single_timestep_part_lue(input_rasters,
                                                                         nonraster_inputs,
                                                                         timestep, ir_info)

        # compute crop areas
        crop_areas =  ans.compute_crop_areas(new_rasters['fracs'], input_rasters['A'],
                                            input_rasters['R'])

        np.save(f'outputs/reg_prod_{timestep}', reg_prod)
        np.save(f'outputs/crop_areas_{timestep}', crop_areas)

        # save difference maps
        _ = ans.compute_diff_rasters(input_rasters['f'], new_rasters['fracs'], timestep-invl,
                                     timestep, save=True)

        # update variables (new_rasters -> input_rasters)
        input_rasters['f'] = new_rasters['fracs']
        input_rasters['lct'] = new_rasters['lct']

@lfr.runtime_scope
def main():
    """main function of new landcover model"""

    rd.check_wdir()
    wt.check_and_create_dirs()

    if not prm.STANDALONE:
        rd.prepare_input_files()
    rd.prepare_input_files()

    start = time()
    input_rasters, nonraster_inputs = setup()

    input_rasters['is_cropland'] = st.isolate_cropland(input_rasters)

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
    ans.compute_largest_fraction(input_rasters['f'], input_rasters['is_cropland'], save=True)

    # compute crop yields from irrigated land
    ir_info = {'ir_yields':compute_irrigated_yield(input_rasters, nonraster_inputs),
               'ir_bool':get_irrigated_boolean(input_rasters),
               'ir_frac':compute_ir_frac(input_rasters)}

    ##############################
    # do the allocation!
    perform_allocation_loop(input_rasters, nonraster_inputs, ir_info, prm.N_STEPS, prm.INTERVAL)
    ##############################

    print(f"Time taken: {time()-start}")

if __name__=="__main__":
    main()
