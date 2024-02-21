"""
Write output files from lue framework to netcdfs.
"""
import xarray as xr
import lue.framework as lfr

def write_raster(raster, var_name, output_dir='outputs/'):
    """
    Writes rasters from LUE into netcdf format.
    
    Parameters
    ----------
    raster : LUE data object
                the raster to be written to disk
    var_name : str
                name of the desired .nc file, excluding the extension
    output_dir : str, default = 'output/'
               relative path to the file
    """

    lfr.wait(raster)
    # lfr.to_gdal(raster, f'{output_dir}{var_name}.nc')
    lfr.to_gdal(raster, f'{output_dir}{var_name}.tif')

def check_format(var_name, data_dir='outputs/'):
    """
    Checks that a netcdf file is in the right format.
    
    Parameters
    ----------
    var_name : str
                name of the .nc file, excluding the extension
    data_dir : str, default = 'output/'
               relative path to the file
    """

    da = xr.open_dataarray(f'{data_dir}{var_name}.nc')
    print(da)

def write_np_raster(file_name, array, output_dir='outputs/'):
    """
    Converts np arrays to lue format and saves as tifs
    """

    shape = array.shape
    # print(f'{file_name} shape: {shape}')

    if len(shape)==3:
        for ind in range(shape[0]):
            lue_array = lfr.from_numpy(array[ind, :, :], partition_shape=shape[1:])
            # prevent GDAL tiff directory count error
            lfr.wait(lue_array)
            lfr.to_gdal(lue_array, f'{output_dir}{file_name}_{ind}.tif')

    elif len(shape)==2:
        lue_array = lfr.from_numpy(array, partition_shape=shape)
        lfr.wait(lue_array)
        lfr.to_gdal(lue_array, f'{output_dir}{file_name}.tif')

    else:
        raise TypeError(f"Array should have rank of 2 or 3, not {len(shape)}")
