import numpy as np
import xarray as xr
import lue.framework as lfr
from read import check_wdir

def quick_write():
    array_np1 = np.random.uniform(size=(4000, 2000))
    array1 = lfr.from_numpy(array_np1, (4000, 2000))

    array_np2 = np.random.uniform(size=(4000, 2000))
    array2 = lfr.from_numpy(array_np2, (4000, 2000))
                            
    array = array1 + array2

    lfr.to_gdal(array, f'outputsuniform.tif')

@lfr.runtime_scope
def main():
    check_wdir()
    quick_write()

    # array = xr.open_dataarray('uniform.nc')
    # print(array)

if __name__=='__main__':
    main()
