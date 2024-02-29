"""
Contains functions to plot saved outputs - both rasters from
np_test_module saved in .npy format and non-raster outputs.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import colormaps as cmap
# from matplotlib.ticker import FormatStrFormatter
import cartopy.crs as ccrs

import read as rd
import parameters as prm

def setup_spatial_plots(n_plots, projection=ccrs.AlbersEqualArea()):
    """
    Sets up required number of figs and axes

    Parameters
    ----------
    n_plots : int
              number of figures to create
    projection : cartopy.crs.CRS object, default = ccrs.AlbersEqualArea()
                 map projection to be used in plots

    Returns
    -------
    plt_tuples : list
                 list of n_plots tuples containing (plt.figure, plt.axes)
                 for each figure created
    """

    plot_tuples = []
    for _ in range(n_plots):
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection=projection)

        ax.gridlines()

        plot_tuples.append((fig, ax))

    return plot_tuples

def plot_raster_like_npy(var_name, nr, nc, shape, projection=ccrs.AlbersEqualArea(),
                         colormap="tab20b"):
    """
    Plots saved outputs from the np_test_module functions

    Parameters
    ----------
    var_name : str
               name of variable to be plotted (eg 'mac' for plot of
               dominant crop type in each cell)
    nr : int
         number of regions
    nc : int
         number of crops
    shape : tuple
            shape of the raster
    projection : cartopy.crs.CRS object, default = ccrs.AlbersEqualArea()
                 map projection to be used in plots
    colormap : str, default = "tab20b"
               string denoting the matplotlib colormap to use in the plots
    """

    # construct meshgrids for plotting
    lons = np.linspace(0, 360, shape[1])
    lats = np.linspace(-90, 90, shape[0])
    lon, lat = np.meshgrid(lons, lats)

    print(shape)
    print(lon.shape)

    var = np.load(f"test_IO\\{var_name}_{nr}_{nc}_{shape}.npy")

    var_shape = var.shape

    # if the variable is a list of maps, then separate it out and make separate plots
    if len(var_shape)==3:
        separated_var = [var[ind, :, :] for ind in var_shape[0]]

        for sv in separated_var:
            # plot preparation
            fig = plt.figure(dpi=150)
            ax = fig.add_subplot(111, projection=projection)

            # plotting
            plotted_var = ax.pcolormesh(lon, lat, sv, shading='nearest',
                                           cmap=mpl.colormaps[colormap])
            plt.colorbar(plotted_var, ax=ax, spacing='uniform')

    # if the variable is 2D, then plot as normal
    elif len(var_shape)==2:
        # plot preparation
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection=projection)

        # plotting
        plotted_var = ax.pcolormesh(lon, lat, var, shading='nearest',
                                           cmap=mpl.colormaps[colormap])
        plt.colorbar(plotted_var, ax=ax, spacing='uniform')

def plot_saved_netcdf(var_name, nr, nc, shape, projection=ccrs.AlbersEqualArea(),
                      colormap="tab20b", data_dir='data'):
    """
    Plots saved outputs from the np_test_module functions

    NB: FUNCTION NOT CURRENTLY WORKING, BUT COULD BE ADAPTED TO PLOT
    NETCDF OUTPUTS WHEN IMAGE-LAND-LUE (not just np_test_module) OUTPUTS
    THESE.
    """

    # construct meshgrids for plotting
    lons = np.linspace(0, 360, shape[1])
    lats = np.linspace(-90, 90, shape[0])
    lon, lat = np.meshgrid(lons, lats)

    print(shape)
    print(lon.shape)

    var = xr.open_dataarray(f"{data_dir}\\{var_name}_{nr}_{nc}_{shape}.nc")

    var_shape = var.shape

    # if the variable is a list of maps, then separate it out and make separate plots
    if len(var_shape)==3:
        separated_var = [var[ind, :, :] for ind in var_shape[0]]

        for sv in separated_var:
            # plot preparation
            fig = plt.figure(dpi=150)
            ax = fig.add_subplot(111, projection=projection)

            # plotting
            plotted_var = ax.pcolormesh(lon, lat, sv, shading='nearest',
                                           cmap=mpl.colormaps[colormap])
            plt.colorbar(plotted_var, ax=ax, spacing='uniform')

    # if the variable is 2D, then plot as normal
    elif len(var_shape)==2:
        # plot preparation
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection=projection)

        # plotting
        plotted_var = ax.pcolormesh(lon, lat, var, shading='nearest',
                                           cmap=mpl.colormaps[colormap])
        plt.colorbar(plotted_var, ax=ax, spacing='uniform')

def load_tab(name, n_steps, invl, zero_start=False):
    """
    Loads and concatenates tabulated data for plotting of timeseries

    Parameters
    ----------
    name : str
           name of the array to be loaded
    n_steps : int
              number of timesteps
    invl : int
           interval between timesteps
    zero_start : bool
                 whether to load initial data, or start from the first
                 model output
    """
    if zero_start:
        strt = 0
    else:
        strt = invl

    arrays = [np.load(f'outputs/{name}_{time}.npy') for time in range(strt, n_steps * invl, invl)]
    tab = np.stack(arrays, axis=0)

    return tab

def define_names():
    """
    Defines label names for all crop types and regions

    RETURNS
    -------
    crops : np.ndarray of type str
            array of crop type names
    regions : np.ndarray of type str
              array of region names
    """
    crops = ['grass', 'wheat', 'rice', 'maize', 'trop. cereals',
                        'oth. temp. cereals', 'pulses', 'soybeans', 'temp. oil crops',
                        'tropical oil crops', 'temp. roots+tubers', 'trop. roots+tubers',
                        'sugar crops', 'oil and palm fruit', 'veg+fruit',
                        'oth. non-food,\n luxury+spices', 'plant-based fibres']

    regions = ['CAN', 'US', 'MEX', 'R. CENT. AMER.', 'BR', 'R. SOU. AMER.',
                        'Northern Africa', 'Western Africa', 'Eastern Africa', 'South Africa',
                        'OECD Europe', 'Eastern Europe', 'Turkey', 'Ukraine +', 'Asia-Stan',
                        'Russia +', 'Middle East', 'India', 'Korea', 'China +', 'South East Asia',
                        'Indonesia +', 'Japan', 'Oceania', 'Rest of S. Asia', 'Rest of S. Africa']

    return crops, regions

def plot_timeseries_overview(tab, crops, regions, invl=None, title=None):
    """
    Plots a set of timeseries together by region and by crop

    Parameters
    ----------
    tab : np.ndarray of type np.float32
          3D array of shape (# timesteps, # crops, # regions)
    crops : np.ndarray of type str
            array of crop type names
    regions : np.ndarray of type str
              array of region names
    invl : int
           interval between timesteps
    """
    # define time array
    if invl is None:
        time = np.arange(tab.shape[0])
    else:
        time = np.arange(tab.shape[0]) * invl + 1970

    # by region
    fig, axes = plt.subplots(4, 7, sharex=True, sharey=True, figsize=(18, 9.5))
    axes = axes.flatten()

    for ind, ax in enumerate(axes):
        if ind==26:
            break
        for crop in range(16):
            ax.plot(time, tab[:, crop, ind], label=crops[crop])
        ax.set_title(regions[ind])

    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])

    if title is not None:
        fig.suptitle(title)

    axes[0].set_zorder(100)
    axes[0].legend(bbox_to_anchor=(8.3, -3.7), loc=4, ncols=2)

    # by crop
    fig2, axes2 = plt.subplots(4, 5, sharex=True, sharey=False, figsize=(18, 9.5))
    axes2 = axes2.flatten()

    for ind, ax in enumerate(axes2):
        if ind==17:
            break
        for reg in range(26):
            ax.plot(time, tab[:, ind, reg], label=regions[reg])
        ax.set_title(crops[ind])

    fig2.delaxes(axes2[-1])
    fig2.delaxes(axes2[-2])
    fig2.delaxes(axes2[-3])

    if title is not None:
        fig2.suptitle(title)

    axes2[0].set_zorder(100)
    axes2[0].legend(bbox_to_anchor=(5.5, -3.7), loc=4, ncols=4)

def plot_timeseries_separately(tab, crops, regions, invl=None, title=None):
    """
    Plots a set of timeseries separately by region and by crop

    Parameters
    ----------
    tab : np.ndarray
    crops : list
    regions : list
    invl = int, default = None
    title = str, default = None
    """
    # define time array
    if invl is None:
        time = np.arange(tab.shape[0])
    else:
        time = np.arange(tab.shape[0]) * invl + 1970

    # by region
    for ind, _ in enumerate(regions):
        fig = plt.figure()
        for crop in range(17):
            plt.plot(time, tab[:, crop, ind], label=f'{crops[crop]}')
        plt.title(f'{title} in {regions[ind]}')
        plt.legend(ncols=2)

def main():
    """main function"""
    rd.check_wdir()

    # uncomment to plot dominant crop type raster output by np_test_module
    # plot_raster_like_npy('mac', 26, 16, (2160, 4320))

    ####### quick demand met test ########
    crop_areas = np.load('outputs/crop_areas_5.npy')
    reg_prod = np.load('outputs/reg_prod_5.npy')

    food_dem = np.load('data/food_crop_demands.npy')
    grass_dem = np.load('data/grass_crop_demands.npy')
    food_dem = np.concatenate((grass_dem[:, 2, np.newaxis, :], food_dem), axis=1)

    print(grass_dem.shape)

    food_prod = np.swapaxes(reg_prod[:, 1:], 0, 1)
    food_dem_1 = food_dem[1, :, :]

    grass_prod = reg_prod[:, 0]
    grass_mismatch = grass_dem[1, 2, :] - grass_prod

    print(grass_prod.shape)
    print(grass_mismatch)
    ######################################

    # define crop types and region names for labels
    crops, regs = define_names()

    # uncomment to plot timeseries of regional crop demands
    # plot_timeseries_overview(food_dem, crops, regs)

    # evolution of crop areas
    crop_area_ts = load_tab('crop_areas', 5, 5, zero_start=True)

    # remove totals, combine rain-fed and irrigated and swap crop and region axes
    crop_area_ts = crop_area_ts[:, :-1, :-1]
    crop_area_ts[:, :, 1:prm.NGFC] += crop_area_ts[:, :, prm.NGFBC:]
    crop_area_ts = crop_area_ts[:, :, :prm.NGFC]
    crop_area_ts = np.swapaxes(crop_area_ts, 1, 2)

    # divide by a million
    crop_area_ts /= 1e6

    # uncomment to plot timeseries of evolution of crop areas
    # plot_timeseries_overview(crop_area_ts, crops, regs, invl=5, title='Crop Area [m. km^2]')
    # plot_timeseries_separately(crop_area_ts, crops, regs, invl=5, title='Cop Area [m. km^2]')

    plt.show()

if __name__=='__main__':
    main()
