"""
Read maps into lue framework, will allocate land.
"""

from time import time

import numpy as np
import pandas as pd
import lue.framework as lfr

import parameters as prm
import read_outfiles as rdo

def prepare_input_files():
    """
    Prepares input files, splitting netcdfs into GDAL-readable formats, etc.
    """

    # read crop demands .OUT files and save them as .npy files to be read in
    rdo.convert_crop_outfiles('food') # pylint: disable=no-member
    rdo.convert_crop_outfiles('grass') # pylint: disable=no-member
    rdo.convert_crop_outfiles('bioenergy') # pylint: disable=no-member

    rdo.convert_management_outfiles('MF') # pylint: disable=no-member
    rdo.convert_management_outfiles('GI') # pylint: disable=no-member
    rdo.convert_management_outfiles('FH') # pylint: disable=no-member

def read_raster(res, file_name, data_dir="data\\"):
    """Reads rasters."""

    if not isinstance(res, int) and not isinstance(res, float):
        raise TypeError("input 'res' must be an integer or float")

    # 1x2 partitions
    res_factor = int(60/res)
    shape = (180*res_factor, 360*res_factor/2)

    path_name = data_dir + file_name

    raster = lfr.from_gdal(path_name, shape) # pylint: disable=no-member

    # ensure datatype is float, not int
    if raster.dtype==np.int32:
        raster = lfr.cast(raster, np.float32)  # pylint: disable=no-member

    return raster

def read_input_rasters(ir_rf=False, data_dir="data\\"):
    """Reads input rasters."""

    # read in constant maps
    greg = read_raster(5, "GREG_5MIN.nc")
    gsuit = read_raster(5, "gsuit_new_world.map")
    garea = read_raster(5, "gareacell.map")

    grmppc_maps = []
    fractions = []
    for crop in range(prm.NFCG): # pylint: disable=no-member
        if not ir_rf:
            # read in crop+1 from gfrac for each crop, because crop 0 is grass (reads IR and RF)
            gfrac_rf = read_raster(5, f"gfrac{crop}.map", data_dir=data_dir+"gfrac\\")
            gfrac_ir = read_raster(5, f"gfrac{crop+21}.map", data_dir=data_dir+"gfrac\\")

            # combine rain-fed and irrigated
            fractions.append(gfrac_rf + gfrac_ir)

            # read in potential productivity maps
            grmppc_rf = read_raster(5, f"grmppc{crop}_5MIN.map", data_dir=data_dir+"grmppc\\")
            grmppc_ir = read_raster(5, f"grmppc{crop+21}_5MIN.map", data_dir=data_dir+"grmppc\\")

            grmppc_maps.append((grmppc_rf+grmppc_ir)/2.)

    return {'R':greg, 'suit':gsuit, 'A':garea, 'p_c':grmppc_maps, 'f':fractions}

def read_nonraster_inputs(ir_rf=False, data_dir="data\\"):
    """Reads (for now) ascii input files"""

    # max productivity for each crop mean of ir and r-f; divided by 10 (kg/ha to tons/km^2)
    max_pr_rf = np.loadtxt(f"{data_dir}MAXPR.txt")[0:prm.NFC] # pylint: disable=no-member
    max_pr_ir = np.loadtxt(f"{data_dir}MAXPR.txt")[22:prm.NFC+22] # pylint: disable=no-member

    if not ir_rf:
        max_pr = np.mean(np.stack([max_pr_rf, max_pr_ir]), axis=0) / 10

    # read in management factor and frharvcomb [ASSUMED CONSTANT!!!]
    m_fac = np.load(f"{data_dir}MF.npy")[0, :, :]
    graz_intens = np.load(f"{data_dir}GI.npy")[0, :, :]
    fr_harv_comb = np.load(f"{data_dir}FH.npy")
    fr_harv_comb = np.ones(prm.NFC) # to make things simpler for now # pylint: disable=no-member


    # load crop demands from .npy files
    food_demands = np.zeros((131, prm.NGFC, 26)) # pylint: disable=no-member
    food_demands[:, 1:, :] = np.load(f"{data_dir}food_crop_demands.npy")
    grass_demands = np.load(f"{data_dir}grass_crop_demands.npy")

    # make total grass demand as the zeroth crop entry of the food_demands array
    food_demands[:, 0, :] = grass_demands[:, 2, :].copy()

    # for _ in range(1000000):
    #     print(grass_demands.shape)
    #     print(f"Length of grass[0, 2, :]: {len(grass_demands[0, 2, :])}")
    #     print(f"Number of elements agreeing: {np.sum(grass_demands[0, 0, :]==food_demands[0, 0, :])}")

    return{'M':max_pr, 'MF':m_fac, 'GI':graz_intens, 'FH':fr_harv_comb, 'd':food_demands}

def setup():
    """Calls the reading functions and prepares the variables for the allocation function."""

    input_rasters = read_input_rasters()
    nonraster_inputs = read_nonraster_inputs()

    # compute regional boolean maps
    greg = input_rasters["R"]
    reg_bools = []
    for reg in range(26):
        reg_bools.append(greg == reg)
    input_rasters["R_bools"] = reg_bools

    return input_rasters, nonraster_inputs

def return_demand_rasters(demands, greg, r_bools, timestep):
    """Takes in demand data and returns rasters for a given timestep, with the 
    demands separated into the different world regions."""

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
    """Takes in demand data and returns rasters of the ratios, in each
    region, of the demand for each crop in this timestep, relative to 
    the last."""

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

def reallocate_cropland_initial(input_rasters, nonraster_inputs):
    """Reallocates all existing cropland, without taking into account whether demand is met."""

    # isolate relevant input variables
    greg = input_rasters["R"]
    reg_bools = input_rasters["R_bools"]
    fractions = input_rasters['f']
    graz_intens = nonraster_inputs['GI']
    demands = nonraster_inputs['d']
    m_fac = nonraster_inputs['MF']

    timestep = 1

    # compute ratio of current demand to previous demand
    demand_ratios = return_demand_ratios(demands, greg, reg_bools, timestep)

    # compute cf_1 and cf_3
    cf1 = fractions[0] * demand_ratios['crop0'] / graz_intens
    cf3 = fractions[0]
    for crop in range(1, 17):
        cf1 += fractions[crop] * demand_ratios[f'crop{crop}'] / m_fac[crop-1]
        cf3 += fractions[crop]

    # compute new fractions from cf_1 and cf_3
    new_fractions = []
    cf_ratio = cf3 / cf1
    for crop in range(1, 17):
        new_fraction = fractions[crop] * m_fac[crop-1] * demand_ratios[f'crop{crop}'] * cf_ratio
        new_fractions.append(new_fraction)

    return new_fractions

@lfr.runtime_scope
def main():
    """main function"""

    if not prm.STANDALONE: # pylint: disable=no-member
        prepare_input_files()
    prepare_input_files()

    start = time()
    # input_rasters, nonraster_inputs = setup()
    # new_fracs = reallocate_cropland_initial(input_rasters, nonraster_inputs)

    print(f"Time taken: {time()-start}")

if __name__=="__main__":
    main()
