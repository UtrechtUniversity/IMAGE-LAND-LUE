"""
Calls the functions in read.py and allocates land.
"""

from time import time

import lue.framework as lfr

import parameters as prm
import read as rd

def setup():
    """
    Calls the read functions and computes regional Boolean rasters.

    Returns
    -------
    input_rasters : Dict
                    dict containing the same items as that returned by
                    read_input_rasters
    nonraster_inputs : Dict
                       dict containing the same items as that returned by
                       read_nonraster_inputs, with the addition of a list
                       of Boolean rasters with values corresponding to
                       each IMAGE World Region

    See Also
    --------
    read_input_rasters
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
            d_map_t = lfr.where(r_bools[reg], local_d_t, d_map_t) # pylint: disable=no-member

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

    for crop in range(17):
        dr_t = greg
        print(f"loading demand for crop {crop}")
        for reg in range(26):
            local_ratio = d_t[crop, reg] / d_tm1[crop, reg]
            # if in correct region, value in every cell is equal to the demand in that region
            dr_t = lfr.where(r_bools[reg], local_ratio, dr_t) # pylint: disable=no-member

        ratio_dict[f"crop{crop}"] = dr_t

    return ratio_dict

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
    read_input_rasters                 
    """
    new_rasters = input_rasters

    # find cropland boolean raster
    is_cropland = input_rasters['lct'] == 1

    # isolate current cropland for every crop fraction
    cc_frs = []
    for crop in range(prm.NGFC):
        cc_frs.append(lfr.where(is_cropland, input_rasters['f'][crop])) # pylint: disable=no-member

    new_rasters['current_cropland'] = cc_frs

    return new_rasters

def get_irrigated_boolean(input_rasters):
    """
    Identifies cells in which there is irrigated cropland.
    
    Parameters
    ----------
    input_rasters : dict
                    dict containing the same items as that returned by
                    read_input_rasters.

    Returns
    -------
    new_rasters : dict
                  dict containing the same items as input_rasters, with 
                  an added Boolean raster showing the extent of irrigated
                  cropland.

    See Also
    --------
    isolate_cropland
    read_input_rasters
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

def reallocate_cropland_initial(input_rasters, nonraster_inputs, timestep=1):
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
    timestep : int, default=1
    
    Returns
    -------
    new_fractions : list of lue data objects
                    list containing rasters for each new crop fraction
    """

    # compute ratio of current demand to previous demand
    demand_ratios = return_demand_ratios(nonraster_inputs['d'], input_rasters["R"],
                                         input_rasters["R_bools"], timestep)

    # relabel relevant input variables
    fractions = input_rasters['current_cropland']
    graz_intens = nonraster_inputs['GI']
    m_fac = nonraster_inputs['MF']

    # compute cf_1 and cf_3
    cf1 = fractions[0] * demand_ratios['crop0'] / graz_intens
    cf3 = fractions[0]
    for crop in range(1, prm.NGFC):
        cf1 += fractions[crop] * demand_ratios[f'crop{crop}'] / m_fac[crop-1]
        cf3 += fractions[crop]

    # compute new fractions from cf_1 and cf_3
    new_fractions = []
    cf_ratio = cf3 / cf1
    for crop in range(1, prm.NGFC):
        new_fraction = fractions[crop] * m_fac[crop-1] * demand_ratios[f'crop{crop}'] * cf_ratio
        new_fractions.append(new_fraction)

    return new_fractions

@lfr.runtime_scope
def main():
    """main function"""

    if not prm.STANDALONE: # pylint: disable=no-member
        rd.prepare_input_files()
    rd.prepare_input_files()

    start = time()
    input_rasters, nonraster_inputs = setup()
    input_rasters = isolate_cropland(input_rasters)
    new_fracs = reallocate_cropland_initial(input_rasters, nonraster_inputs)

    print(f"Time taken: {time()-start}")

if __name__=="__main__":
    main()
