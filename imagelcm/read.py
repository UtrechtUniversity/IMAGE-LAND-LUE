"""
Read input files into the lue framework.
"""

import os
import numpy as np
import lue.framework as lfr

import parameters as prm
import convert_outfiles as co

def check_wdir():
    """
    Ensures that the current working directory is correct.
    """

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

def prepare_input_files():
    """
    Converts .out files to .npy format and makes netcdfs GDAL-readable.

    See Also
    --------
    convert_crop_outfiles
    convert_management_outfiles
    """

    # read crop demands .OUT files and save them as .npy files to be read in
    co.convert_crop_outfiles('food') # pylint: disable=no-member
    co.convert_crop_outfiles('grass') # pylint: disable=no-member
    co.convert_crop_outfiles('bioenergy') # pylint: disable=no-member

    co.convert_management_outfiles('MF') # pylint: disable=no-member
    co.convert_management_outfiles('GI') # pylint: disable=no-member
    co.convert_management_outfiles('FH') # pylint: disable=no-member

def read_raster(res, file_name, data_dir="data\\"):
    """
    Reads rasters of specified resolution into LUE.
    
    Parameters
    ----------
    res : int or float
          resolution of the raster to be read in
    file_name : str
                name of the raster file, inclusding the extension
    data_dir : str, default = 'data\\'
               relative path to the file
    
    Returns
    -------
    raster : lue data object

    Raises
    ------
    TypeError
        if res is neither an int nor a float
    """

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
    """
    Reads IMAGE-LAND's raster inputs and returns them as a dictionary.
    
    Parameters
    ----------
    ir_rf : Bool, default=False
            whether the LCM handles irrigated cropland
    data_dir : string, default = 'data\\'
               relative path to the file

    Returns
    -------
    input_rasters : Dict
                    dict containing the map of regions, suitability 
                    map, grid area raster, a list containing grid-reduced
                    pot. productivity maps, a list containting the crop
                    fractions and the land-cover type raster
    """

    # read in constant maps
    greg = read_raster(5, "GREG_5MIN.nc")
    glct = read_raster(5, "GLCT.NC")
    gsuit = read_raster(5, "gsuit_new_world.map")
    garea = read_raster(5, "gareacell.map")

    grmppc_maps = []
    fractions = []
    for crop in range(prm.NGFC): # pylint: disable=no-member
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

    input_rasters = {'R':greg, 'suit':gsuit, 'A':garea, 'p_c':grmppc_maps, 'f':fractions,
                     'lct':glct}

    return input_rasters

def read_nonraster_inputs(ir_rf=False, data_dir="data\\"):
    """
    Reads IMAGE-LAND's non-raster inputs and returns them as a dictionary.

    Parameters
    ----------
    ir_rf : Bool, default=False
            whether the LCM handles irrigated cropland
    data_dir : string, default = 'data\\'
               relative path to the file

    Returns
    -------
    nonraster_inputs : Dict
                        dict containing a np array containing the
                        maximum potential productivity for all crops, a np
                        array containing the management factor for all
                        crops for all regions and timesteps, a np array
                        containing the grazing intensities in all regions
                        and timesteps, a np array containing the harvest
                        fraction, for all crops, regions and timesteps and
                        a np array containing the demands for food and
                        fodder crops for all regions and timesteps.
    """

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

    #for _ in range(1000000):
    #print(grass_demands.shape)
    #print(f"Length of grass[0, 2, :]: {len(grass_demands[0, 2, :])}")
    #print(f"Number of elements agreeing: {np.sum(grass_demands[0, 0, :]==food_demands[0, 0, :])}")

    nonraster_inputs = {'M':max_pr, 'MF':m_fac, 'GI':graz_intens, 'FH':fr_harv_comb,
                         'd':food_demands}

    return nonraster_inputs
