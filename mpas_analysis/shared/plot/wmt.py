# This software is open source software available under the BSD-3 license.
#
# Copyright (c) 2022 Triad National Security, LLC. All rights reserved.
# Copyright (c) 2022 Lawrence Livermore National Security, LLC. All rights
# reserved.
# Copyright (c) 2022 UT-Battelle, LLC. All rights reserved.
#
# Additional copyright and license information can be found in the LICENSE file
# distributed with this code, or at
# https://raw.githubusercontent.com/MPAS-Dev/MPAS-Analysis/main/LICENSE
"""
Functions for plotting surface water mass transformation
"""
# Authors
# -------
# Carolyn Begeman

import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np

from mpas_analysis.shared.timekeeping.utility import date_to_days

from mpas_analysis.shared.constants import constants

from mpas_analysis.shared.plot.ticks import plot_xtick_format
from mpas_analysis.shared.plot.title import limit_title


def wmt_yearly_plot(config, dsBins, dsValues,
                    mode='cumulative',
                    title=None, xLabel=None, yLabel=None,
                    lineColors=None, lineStyles=None, markers=None,
                    lineWidths=None, legendText=None,
                    titleFontSize=None, axisFontSize=None,
                    defaultFontSize=None, figsize=(12, 6), dpi=None,
                    legendLocation='best', maxTitleLength=90):

    """
    Plots the list of histogram data sets.

    Parameters
    ----------
    config : instance of ConfigParser
        the configuration, containing a [plot] section with options that
        control plotting

    dsBins: xArray DataSet
        the data set containing the density bin centers and edges

    dsValues : list of xarray DataSets
        the data set(s) to be plotted. Datasets should already be sliced
        within the time range specified in the config file.

    title : str
        the title of the plot

    xLabel, yLabel : str
        axis labels

    lineColors, lineStyles, legendText : list of str, optional
        control line color, style, and corresponding legend
        text.  Default is black, solid line, and no legend.

    lineWidths : list of float, optional
        control line width.  Default is 1.0.

    titleFontSize : int, optional
        the size of the title font

    defaultFontSize : int, optional
        the size of text other than the title

    figsize : tuple of float, optional
        the size of the figure in inches

    dpi : int, optional
        the number of dots per inch of the figure, taken from section ``plot``
        option ``dpi`` in the config file by default

    legendLocation : str, optional
        The location of the legend (see ``pyplot.legend()`` for details)

    maxTitleLength : int, optional
        the maximum number of characters in the title and legend, beyond which
        they are truncated with a trailing ellipsis

    Returns
    -------
    fig : ``matplotlib.figure.Figure``
        The resulting figure
    """
    # Authors
    # -------
    # Carolyn Begeman, Adrian Turner, Xylar Asay-Davis

    if defaultFontSize is None:
        defaultFontSize = config.getint('plot', 'defaultFontSize')
    matplotlib.rc('font', size=defaultFontSize)

    if dpi is None:
        dpi = config.getint('plot', 'dpi')

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if title is not None:
        if titleFontSize is None:
            titleFontSize = config.get('plot', 'titleFontSize')
        title_font = {'size': titleFontSize,
                      'color': config.get('plot', 'titleFontColor'),
                      'weight': config.get('plot', 'titleFontWeight')}
    if axisFontSize is None:
        axisFontSize = config.get('plot', 'axisFontSize')
    axis_font = {'size': axisFontSize}

    # get density bin centers
    bin_centers = ds.densBinsCenters.isel(time=0)

    # get density variables from datasets
    if mode == 'cumulative':
        data_vars = ['dens_IOAO_FWflux', 'dens_hap']
    elif mode == 'decomp':
        data_vars = ['dens_AO_FWflux', 'dens_rivRof', 'dens_ISMF', 'dens_melt',
                     'dens_brine', 'dens_iceRof']
    else:
        raise(f'Mode {mode} not supported')

    ds = xr.Dataset()
    for ds_single in dsValues:
        for data_var in data_vars:
            if data_var in ds.keys():
                ds = xr.concat([ds, ds_single], dim='time',
                               data_vars={data_var},
                               coords='minimal')
            else:
                ds[data_var] = ds_single[data_var]

    if title is not None:
        title = limit_title(title, maxTitleLength)
        if titleFontSize is None:
            titleFontSize = config.get('plot', 'titleFontSize')
        title_font = {'size': titleFontSize,
                      'color': config.get('plot', 'titleFontColor'),
                      'weight': config.get('plot', 'titleFontWeight')}
    if axisFontSize is None:
        axisFontSize = config.get('plot', 'axisFontSize')
    axis_font = {'size': axisFontSize}
    if markers is None or len(markers) != len(data_vars):
        markers = [None for i in data_vars]
    if lineStyles is None or len(lineStyles) != len(data_vars):
        lineStyles = ['-' for i in data_vars]
    if lineWidths is None or len(lineWidths) != len(data_vars):
        lineWidths = [1.0 for i in data_vars]
    if mode == 'cumulative':
        ds['dens_total'] = ds.dens_IOAO_FWflux + ds.dens_hap
        data_vars.append = 'dens_total'
        if lineColors is None or len(lineColors) != len(data_vars):
            lineColors = [TODO]
        if legendText is None or len(legendText) != len(data_vars):
            legendText = [TODO]
    elif mode == 'decomp':
        ds['dens_EPR'] = ds.dens_AO_FWflux + ds.dens_rivRof
        data_vars.append = 'dens_EPR'
        if lineColors is None or len(lineColors) != len(data_vars):
            lineColors = [TODO]
        if legendText is None or len(legendText) != len(data_vars):
            legendText = [TODO]

    if xLabel is None:
        xLabel = r'Neutral density, $\gamma_n$ ($kg\:m^{-3}$)'
    if yLabel is None:
        yLabel = 'Transformation rate (Sv)'

    fig = plt.figure(figsize=figsize, dpi=dpi)
    for idx, data_var in data_vars:
        data = ds[data_var]
        data = data.mean(dim='time')
        plot(bin_centers, data,
             color=lineColors[idx], linestyle=lineStyles[idx],
             linewidth=lineWidths[idx], marker=markers[idx],
             label=limit_title(legendText[idx], maxTitleLength))
    if title is not None:
        plt.title(title, **title_font)
    if title is not None:
        if titleFontSize is None:
            titleFontSize = config.get('plot', 'titleFontSize')
        title_font = {'size': titleFontSize,
                      'color': config.get('plot', 'titleFontColor'),
                      'weight': config.get('plot', 'titleFontWeight')}
    if axisFontSize is None:
        axisFontSize = config.get('plot', 'axisFontSize')
    axis_font = {'size': axisFontSize}
    plt.xlabel(xLabel, **axis_font)
    plt.ylabel(yLabel, **axis_font)
    plt.legend(loc=LegendLocation)

    return fig 
