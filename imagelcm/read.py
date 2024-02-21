"""
Read input files into the lue framework.
"""

import os
import numpy as np
import xarray as xr
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

def read_raster(res, file_name, data_dir="data/"):
    """
    Reads rasters of specified resolution into LUE.
    
    Parameters
    ----------
    res : int or float
          resolution of the raster to be read in
    file_name : str
                name of the raster file, including the extension
    data_dir : str, default = 'data/'
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

    raster = lfr.from_gdal(path_name, partition_shape=shape) # pylint: disable=no-member

    # ensure datatype is float, not int
    if raster.dtype==np.int32:
        raster = lfr.cast(raster, np.float32)  # pylint: disable=no-member

    return raster

def read_raster_np(res, file_name, data_dir="data/", target_dtype='float'):
    """
    Reads rasters of specified resolution into LUE VIA XARRAY AND NUMPY!
    
    Parameters
    ----------
    res : int or float
          resolution of the raster to be read in
    file_name : str
                name of the raster file, including the extension
    data_dir : str, default = 'data/'
               relative path to the file
    target_dtype : str, default = 'float'
                   desired datatype of the values in the returned raster
    
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

    datarray = xr.open_dataarray(path_name)
    raster_np = datarray.values
    rank = len(raster_np.shape)

    # if rank of array = 3, time axis must be removed
    if rank==3:
        raster_np = raster_np[0, :, :]

    # raster = lfr.from_gdal(path_name, shape) # pylint: disable=no-member
    raster = lfr.from_numpy(raster_np, partition_shape=shape) # pylint: disable=no-member

    # ensure datatype is float, not int
    if target_dtype=='float' and raster.dtype==np.int32:
        raster = lfr.cast(raster, np.float32)  # pylint: disable=no-member
    elif target_dtype=='int' and raster.dtype!=np.int32:
        raster = lfr.cast(raster, np.int32)  # pylint: disable=no-member
    elif target_dtype not in ('float', 'int'):
        raise ValueError("target_dtype must have a string value of <int> or <float>")

    return raster

def read_input_rasters(data_dir="data/"):
    """
    Reads IMAGE-LAND's raster inputs and returns them as a dictionary.
    
    Parameters
    ----------
    data_dir : string, default = 'data/'
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
    # greg = read_raster_np(5, "GREG_5MIN.nc", target_dtype='int')
    greg = read_raster_np(5, "greg_5min_int.nc", target_dtype='int')
    glct = read_raster_np(5, "GLCT.NC")
    gsuit = read_raster_np(5, "gsuit_new_world.nc")
    # garea = read_raster_np(5, "gareacell.nc")
    garea = read_raster_np(5, "GAREACELLNOWATER.nc")

    grmppc_maps = []
    fractions = []
    for crop in range(prm.NGFBFC): # pylint: disable=no-member
        fractions.append(read_raster_np(5, f"gfrac{crop}.nc", data_dir=data_dir+"gfrac/"))
        grmppc_maps.append(read_raster(5, f"grmppc{crop}_5MIN.tif", data_dir=data_dir+"grmppc/"))

    input_rasters = {'R':greg, 'suit':gsuit, 'A':garea, 'p_c':grmppc_maps, 'f':fractions,
                     'lct':glct}

    return input_rasters

def read_nonraster_inputs(data_dir="data/"):
    """
    Reads IMAGE-LAND's non-raster inputs and returns them as a dictionary.

    Parameters
    ----------
    data_dir : string, default = 'data/'
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
    max_pr = np.loadtxt(f"{data_dir}MAXPR.txt") / 10

    # read in management factor and frharvcomb
    m_fac = np.load(f"{data_dir}MF.npy")
    graz_intens = np.load(f"{data_dir}GI.npy")
    fr_harv_comb = np.load(f"{data_dir}FH.npy")

    # load crop demands from .npy files
    food_demands = np.zeros((131, prm.NGFC, 26)) # pylint: disable=no-member
    food_demands[:, 1:, :] = np.load(f"{data_dir}food_crop_demands.npy")
    grass_demands = np.load(f"{data_dir}grass_crop_demands.npy")

    # make total grass demand as the zeroth crop entry of the food_demands array
    food_demands[:, 0, :] = grass_demands[:, 2, :].copy()

    nonraster_inputs = {'M':max_pr, 'd':food_demands, 'GI_all':graz_intens, 'MF_all':m_fac,
                        'FH_all':fr_harv_comb}

    return nonraster_inputs
