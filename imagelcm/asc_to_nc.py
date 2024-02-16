"""Converts rasters saved in ASCII format to netcdfs"""

import numpy as np
import xarray as xr
from read import check_wdir

def convert_asc_nc(file_name, rel_path="data/"):
    """
    Converts a file rel_path/file_name.asc from ASCII to netcdf format
    """

    np_arr = np.loadtxt(f'{rel_path}{file_name}.asc', skiprows=6)

    # pre-processing
    np_arr[np_arr<0] = np.nan
    np_arr = np_arr.astype(np.float32)

    # take netcdf shape from gsuit.nc
    gsuit = xr.open_dataarray('data/gsuit_new_world.nc')
    arr = gsuit.mean('time')
    arr.values = np_arr
    # greg.rename({'gsuit_new_world':'greg'})
    arr.name = file_name
    print(arr)

    # save to file
    arr.to_netcdf(f'{rel_path}{file_name}.nc')

check_wdir()
convert_asc_nc('GAREACELLNOWATER')
