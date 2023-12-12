"""
Plots saved outputs (eventually in both .npy and .nc formats)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs

import read as rd

def setup_spatial_plots(n_plots, projection=ccrs.AlbersEqualArea()):
    """
    Sets up required number of figs and axes
    """

    plot_tuples = []
    for _ in range(n_plots):
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection=projection)

        ax.gridlines()

        plot_tuples.append((fig, ax))

    return plot_tuples

def plot_saved_npy(var_name, nr, nc, shape, projection=ccrs.AlbersEqualArea(), colormap='rainbow'):
    """
    Plots saved outputs from the np_test_module functions
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

rd.check_wdir()
# plt_tups = setup_spatial_plots(1)
# print(plt_tups)
plot_saved_npy('mac', 26, 16, (2160, 4320))

plt.show()
