"""
Write output files and ensure correct relative file paths.
"""
import os
import xarray as xr
import lue.framework as lfr
import read as rd

def create_folder(dir_name, rel_path=""):
    """
    Checks for existance of a directory and creates it if necessary

    PARAMETERS
    ----------
    dir_name : str
               name of the directory to be created (if necessary)
    rel_path : str, default = ""
               path between current working directory and target directory
    """
    # Get the current working directory
    current_directory = os.getcwd()

    # Create the full path of the folder
    folder_path = os.path.join(current_directory, rel_path, dir_name)

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # If not, create the folder
        os.makedirs(folder_path)
        print(f"Folder '{dir_name}' created successfully.")
    else:
        print(f"Folder '{dir_name}' already exists.")

def check_and_create_dirs():
    """
    Checks for the existance of and creates direcories if necessary
    """
    rd.check_wdir()

    # output directories
    create_folder('outputs')
    create_folder('diff_rasters', 'outputs')
    create_folder('frac_rasters', 'outputs')
    create_folder('test_IO')

    # documentation build directory
    create_folder('build', '../docs')

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
