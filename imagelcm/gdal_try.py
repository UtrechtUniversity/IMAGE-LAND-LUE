from osgeo import gdal
import sys
from read import check_wdir

check_wdir()

# register all of the drivers
gdal.AllRegister()

fn = 'data\\gfrac\\gfrac0.nc'
ds = gdal.Open(fn, gdal.GA_ReadOnly)
if ds is None:
    print('Could not open ' + fn)
    sys.exit(1)
