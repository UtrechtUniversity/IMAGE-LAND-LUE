# Inputs

The inputs to IMAGE-LAND-LUE are as follows.
* Rasters:
	* Regions map, constant
	* Suitability map, constant
	* Grid-reduced potential productivity (GRMPPC), one raster per year per crop
	* Initial land cover types
	* Initial fractions (year 1970)
	* Area, excluding water and urban areas, constant
* Non-raster inputs:
	* Crop demands (one value per year, per crop and region)
		* <a href='Food Crops.html'>Food Crops</a>
		* <a href='Grass.html'>Grass</a>
		* Bioenergy
	*  <a href='Management Factor.html'>Management Factor</a>
	* Grazing intensity (see <a href='Grass.html'>grass</a>), one value per year, region
	* FHarvComb, the fraction of allocated land harvested each year (one value per year, per crop and region).

All raster inputs are read in from netcdf format. These are read into the lue framework via lue's
from_numpy() function: they are first read into the model using xarray, then converted to numpy format,
from which they can be converted into lue partitioned arrays.

Non-raster inputs were initially stored using the IMAGE model specific .OUT files (in binary format).
Convert_outfiles.py converts these into numpy's <a href='https://numpy.org/devdocs/reference/generated/numpy.lib.format.html'>npy</a>
format, so they can more readily be manipulated as numpy.ndarrays.