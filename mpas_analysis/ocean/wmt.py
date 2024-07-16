# -*- coding: utf-8 -*-
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
#
import os
from fastjmd95 import rho, drhodt, drhods
import glob
import xarray
import numpy
import matplotlib.pyplot as plt

from mpas_analysis.shared import AnalysisTask

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf_with_fill
from mpas_analysis.shared.io.utility import build_config_full_path, \
    build_obs_path, make_directories, decode_strings
from mpas_analysis.shared.climatology import compute_climatology, \
    get_unmasked_mpas_climatology_file_name

from mpas_analysis.shared.constants import constants
from mpas_analysis.shared.plot import wmt_yearly_plot, savefig
from mpas_analysis.shared.html import write_image_xml

class OceanWMT(AnalysisTask):
    """
    Plots WMT in an ocean region.
    """
    def __init__(self, config, regionMasksTask,
                 controlConfig=None):

        """
        Construct the analysis task.

        Parameters
        ----------
        config : mpas_tools.config.MpasConfigParser
            Configuration options

        regionMasksTask : ``ComputeRegionMasks``
            A task for computing region masks

        controlConfig : mpas_tools.config.MpasConfigParser
            Configuration options for a control run (if any)
        """

        # first, call the constructor from the base class (AnalysisTask)
        super().__init__(
            config=config,
            taskName='oceanWMT',
            componentName='ocean',
            tags=['climatology', 'regions', 'wmt'])

        self.controlConfig = controlConfig
        mainRunName = config.get('runs', 'mainRunName')

        self.regionGroups = config.getexpression(self.taskName, 'regionGroups')
        self.regionNames = config.getexpression(self.taskName, 'regionNames')

        startYear = config.getint(self.taskName, 'startYear')
        endYear = config.getint(self.taskName, 'endYear')
        seasons = config.getexpression(self.taskName, 'seasons')
        densityBins = config.getexpression(self.taskName, 'densityBins',
                                           use_numpyfunc=True)
        fluxVariables = config.getexpression(self.taskName, 'fluxVariables')

        if controlConfig is not None:
            controlRunName = controlConfig.get('runs', 'mainRunName')
            galleryName = None
            refTitleLabel = 'Control: {}'.format(controlRunName)
            outFileLabel = 'wmt'
            diffTitleLabel = 'Main - Control'

        baseDirectory = build_config_full_path(
            config, 'output', 'wmtSubdirectory')
        if not os.path.exists(baseDirectory):
            make_directories(baseDirectory)

        years = list(range(startYear, endYear + 1))

        # run one mask subtask per region group
        for regionGroup in self.regionGroups:

            mpasMasksSubtask = regionMasksTask.add_mask_subtask(
                regionGroup=regionGroup)
            regionNames = mpasMasksSubtask.expand_region_names(
                self.regionNames)

            regionGroupSuffix = regionGroup.replace(' ', '_')

            # run one compute and one plot subtask per region name per year
            for regionName in regionNames:

                regionNameSuffix = regionGroup.replace(' ', '_')
                filePrefix = f'wmt_{regionGroupSuffix}_{regionNameSuffix}'

                for year in years:

                    # Generate wmt fields for each region
                    computeRegionSubtask = ComputeRegionWmtSubtask(
                        self, regionGroup, regionName,
                        startYear=year, endYear=year,
                        masksSubtask=mpasMasksSubtask,
                        densityBins=densityBins,
                        filePrefix=filePrefix)
                    computeRegionSubtask.run_after(mpasMasksSubtask)
                    self.add_subtask(computeRegionSubtask)

                # TODO ClimoRegionSubtask (optional)
                # -- takes flux variables with dimensions (nCellsRegion, nTime, nBins) and performs mean over Time to collapse that dimension
                # -- this is only useful if we want to be able to plot the WMT map for a given density class
                # climoRegionSubtask.run_after(computeRegionSubtask)

                # Generate one figure of wmt vs. density for each season
                for season in seasons:
                    plotRegionSubtask = PlotRegionWmtSubtask(
                        self, regionGroup, regionName, controlConfig,
                        sectionName, filePrefix, mpasMasksSubtask)
                    plotRegionSubtask.run_after(computeRegionSubtask)
                    self.add_subtask(plotRegionSubtask)

    def setup_and_check(self):
        """
        Perform steps to set up the analysis and check for errors in the setup.

        Raises
        ------
        OSError
            If files are not present
        """
        # first, call setup_and_check from the base class (AnalysisTask),
        # which will perform some common setup, including storing:
        #   self.inDirectory, self.plotsDirectory, self.namelist, self.streams
        #   self.calendar
        super().setup_and_check()

        # get a list of timeSeriesStats output files from the streams file,
        # reading only those that are between the start and end dates

class ComputeRegionWmtSubtask(AnalysisTask):
    """
    Computes density bin masks for each time in range from T,S for a particular region
    Fetches all flux variables for that region for each time in range
    Applies density bin masks to each flux variable
    (a) Saves a netcdf file for each region with masked flux variables with dimensions (nCellsRegion, nTime, nBins)
        and density bin variables with dimensions (nBins)
    or
    (b) Takes flux variables with dimensions (nCellsRegion, nTime, nBins)
        and performs mean over Time to collapse that dimension
        then save that to a netcdf file
    """
    def __init__(self, parentTask, regionGroup, regionName, startYear, endYear,
                 masksSubtask, densityBins, fluxVariables, filePrefix):

        super(ComputeRegionWmtSubtask, self).__init__(
            config=parentTask.config,
            taskName=parentTask.taskName,
            componentName=parentTask.componentName,
            tags=parentTask.tags,
            subtaskName=f'computeWmt_{regionGroup}_{regionName}_'
                        f'{startYear:04d}-{endYear:04d}')

        self.masksSubtask = masksSubtask
        self.filePrefix = f'{filePrefix}_{startYear:04d}-{endYear:04d}'
        self.run_after(masksSubtask)
        self.startYear = startYear
        self.endYear = endYear
        self.densityBins = densityBins
        self.fluxVariables = fluxVariables
        self.startDate = f'{self.startYear:04d}-01-01_00:00:00'
        self.endDate = f'{self.endYear:04d}-12-31_23:59:59'

    def setup_and_check(self):
        """
        Perform steps to set up the analysis and check for errors in the setup.

        Raises
        ------
        IOError
            If an input file is not present
        """
        super(ComputeRegionWmtSubtask, self).setup_and_check()

        self.streamName = 'timeSeriesStatsMonthlyOutput'
        self.inputFiles = self.historyStreams.readpath(
            self.streamName, startDate=self.startDate, endDate=self.endDate,
            calendar=self.calendar)

        if len(self.inputFiles) == 0:
            raise IOError('No files were found in stream {} between {} and '
                          '{}.'.format(self.streamName, self.startDate,
                                       self.endDate))

        self.variableList = ['timeMonthly_avg_activeTracers_temperature',
                             'timeMonthly_avg_activeTracers_salinity']
        for variable in self.fluxVariables:
            self.variableList.append(f'timeMonthly_avg_{variable}')

    def run_task(self):

        iselValues = {'Time': 0,
                      'nVertLevels': 0}
        surfacePressure = 0. # dbar

        nBins = len(densityBins)

        nCells = ds.sizes('nCells')

        # Loop over time by loading inputFiles
        for inputFile in inputFiles:

            ds = open_mpas_dataset(fileName=inputFile,
                                   calendar=self.calendar,
                                   variableList=self.variableList,
                                   startDate=self.startDate,
                                   endDate=self.endDate)

            ds = ds.isel(iselValues)

            ds_new = xr.Dataset()
            ds_new['xTime'] = ds.xTime

            # Compute density from T,S
            temperature = ds['timeMonthly_avg_activeTracers_temperature']
            salinity = ds['timeMonthly_avg_activeTracers_salinity']
            density = rho(salinity, temperature, surface_pressure)
            # Loop over density bins
            for i, density_min in enumerate(densityBins):
        
                density_max = densityBins[i+1]
                # Compute density bin mask
                density_mask = xr.logical_and(density >= density_min,
                                              density < density_max) 
                # Loop over flux variables
                flux_masked = np.zeros((nBins, nCells))
                for variable in self.fluxVariable:
                    outputFile = f'{filePrefix}_{variable}'

                    # Apply density bin mask to flux variable
                    flux_masked[i, :] = ds[f'timeMonthly_avg_{variable}'].values * mask
                    ds_new['timeMonthly_avg_{variable}'] = flux_masked
            ds_out = xr.concat([ds_out, ds_new], dim='Time')
        # Loop over density bins
        #    Loop over flux variables
        #        apply time averaging
        #        save netcdf
    
class PlotRegionWmtSubtask(AnalysisTask):
    """
    Plots a histogram diagram for a given ocean region

    Attributes
    ----------
    regionGroup : str
        Name of the collection of region to plot

    regionName : str
        Name of the region to plot

    sectionName : str
        The section of the config file to get options from

    controlConfig : mpas_tools.config.MpasConfigParser
        The configuration options for the control run (if any)

    mpasMasksSubtask : ``ComputeRegionMasksSubtask``
        A task for creating mask MPAS files for each region to plot, used
        to get the mask file name

    """

    def __init__(self, parentTask, regionGroup, regionName, controlConfig,
                 sectionName, fullSuffix, mpasMasksSubtask):

        """
        Construct the analysis task.

        Parameters
        ----------
        parentTask :  ``AnalysisTask``
            The parent task, used to get the ``taskName``, ``config`` and
            ``componentName``

        regionGroup : str
            Name of the collection of region to plot

        regionName : str
            Name of the region to plot

        controlconfig : mpas_tools.config.MpasConfigParser, optional
            Configuration options for a control run (if any)

        sectionName : str
            The config section with options for this regionGroup

        fullSuffix : str
            The regionGroup and regionName combined and modified to be
            appropriate as a task or file suffix

        mpasMasksSubtask : ``ComputeRegionMasksSubtask``
            A task for creating mask MPAS files for each region to plot, used
            to get the mask file name
        """

        # first, call the constructor from the base class (AnalysisTask)
        super(PlotRegionWmtSubtask, self).__init__(
            config=parentTask.config,
            taskName=parentTask.taskName,
            componentName=parentTask.componentName,
            tags=parentTask.tags,
            subtaskName=f'plot_wmt_{fullSuffix}_{regionName}')

        self.regionGroup = regionGroup
        self.regionName = regionName
        self.sectionName = sectionName
        self.controlConfig = controlConfig
        self.mpasMasksSubtask = mpasMasksSubtask
        self.filePrefix = fullSuffix

    def setup_and_check(self):
        """
        Perform steps to set up the analysis and check for errors in the setup.

        Raises
        ------
        IOError
            If files are not present
        """

        # first, call setup_and_check from the base class (AnalysisTask),
        # which will perform some common setup, including storing:
        #   self.inDirectory, self.plotsDirectory, self.namelist, self.streams
        #   self.calendar
        super(PlotRegionWmtSubtask, self).setup_and_check()

        self.xmlFileNames = []
        for var in self.variableList:
            self.xmlFileNames.append(
                f'{self.plotsDirectory}/{self.filePrefix}_'
                f'{self.regionName}.xml')

    def run_task(self):
        """
        Plots WMT in an ocean region.
        """

        self.logger.info(f"\nPlotting WMT for "
                         f"{self.regionName}")

        config = self.config

        main_run_name = config.get('runs', 'mainRunName')

        base_directory = build_config_full_path(
            config, 'output', 'wmtSubdirectory')

        # This bit of code assumes we are saving separate files for each year
        # filenames = glob.glob(
        #         f'{base_directory}/{self.filePrefix}_{self.regionName}_*' \
        #         'wmt.nc')

        # ds_bins = filenames[0]
        # ds_values = []
        # for filename in filenames:
        #     with xr.open_dataset(filename) as ds:
        #         ds_values.append(ds)

        # TODO
        # open single file corresponding to the region and whole time window considered
        # average over nCellsRegion to get ds_values

        wmt_yearly_plot(config, ds_bins, ds_values, mode='cumulative')
        file_prefix = f'{self.filePrefix}_' \
                       'cumulative_surface_flux_' \
                       f'{self.regionName}_wmt'
        caption = f'Climatological surface flux WMT for ' \
                  f'{self.regionName.replace("_", " ")}'
        out_filename = f'{self.plotsDirectory}/{file_prefix}.png'
        savefig(out_filename, config)

        write_image_xml(
            config=config,
            filePrefix=file_prefix,
            componentName='Ocean',
            componentSubdirectory='ocean',
            galleryGroup=f'{self.regionGroup} WMT',
            groupLink=f'wmtCumulative',
            gallery='wmt',
            thumbnailDescription=f'{self.regionName.replace("_", " ")} ',
            imageDescription=caption,
            imageCaption=caption)
