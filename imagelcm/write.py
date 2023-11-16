"""
Write output files from lue framework to netcdfs.
"""
import xarray as xr
import lue.framework as lfr

def write_raster(raster, var_name, output_dir='output\\'):
    """
    Writes rasters from LUE into netcdf format.
    
    Parameters
    ----------
    raster : LUE data object
                the raster to be written to disk
    var_name : str
                name of the desired .nc file, excluding the extension
    output_dir : str, default = 'output\\'
               relative path to the file
    """

    lfr.to_gdal(raster, f'{output_dir}{var_name}.nc')

def check_format(var_name, data_dir='output\\'):
    """
    Checks that a netcdf file is in the right format.
    
    Parameters
    ----------
    var_name : str
                name of the .nc file, excluding the extension
    data_dir : str, default = 'output\\'
               relative path to the file
    """

    da = xr.open_dataarray(f'{data_dir}{var_name}.nc')
    print(da)
