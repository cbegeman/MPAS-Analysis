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
import glob
import xarray
import numpy
import matplotlib.pyplot as plt
from .jmd95wrapper import rho, drhodt, drhods

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

        sectionName = self.taskName
        self.regionGroups = config.getexpression(sectionName, 'regionGroups')
        self.regionNames = config.getexpression(sectionName, 'regionNames')

        startYear = config.getint(self.taskName, 'startYear')
        endYear = config.getint(self.taskName, 'endYear')
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

        # run one mask subtask per region group
        for regionGroup in self.regionGroups:

            mpasMasksSubtask = regionMasksTask.add_mask_subtask(
                regionGroup=regionGroup)
            regionNames = mpasMasksSubtask.expand_region_names(
                self.regionNames)
            print(f'init maskFileName as {mpasMasksSubtask.maskFileName}')

            # run one compute and one plot subtask per region name per year
            # TODO is there a problem with multiple tasks corresponding to different regions accessing the same timeMonthly files?
            # For now we run all regionNames in a region group together
            # for regionName in regionNames:

                # Generate wmt fields for each region
                # -- The output of this is a saved dataset for each region and year
                #    TODO consider if we really want to save every year or just accumulate the time average in one file (possibly for each season)
                # -- The dataset has dimensions (nCellsRegion, nBins) with flux variables
            for year in range(startYear, endYear + 1):
                computeRegionSubtask = ComputeRegionWmtSubtask(
                    self, regionGroup, regionNames,
                    startYear=year, endYear=year,
                    masksSubtask=mpasMasksSubtask,
                    densityBins=densityBins,
                    fluxVariables=fluxVariables)
                computeRegionSubtask.run_after(mpasMasksSubtask)
                self.add_subtask(computeRegionSubtask)

                # TODO ClimoRegionSubtask (optional)
                # -- loads flux variables with dimensions (nCellsRegion, nBins) for each year
                #    -- performs mean over Time to collapse that dimension
                # -- this is only useful if we want to be able to plot the WMT map for a given density class
                # climoRegionSubtask.run_after(computeRegionSubtask)

            # Generate one figure of wmt vs. density for each region
            for regionName in self.regionNames:
                # seasons not yet supported
                plotRegionSubtask = PlotRegionWmtSubtask(
                    self, regionGroup, regionName, controlConfig,
                    sectionName, mpasMasksSubtask,
                    fluxVariables=fluxVariables)
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
    def __init__(self, parentTask, regionGroups, regionNames, startYear, endYear,
                 masksSubtask, densityBins, fluxVariables):

        super(ComputeRegionWmtSubtask, self).__init__(
            config=parentTask.config,
            taskName=parentTask.taskName,
            componentName=parentTask.componentName,
            tags=parentTask.tags,
            subtaskName=f'computeWmt_{startYear:04d}-{endYear:04d}')

        self.masksSubtask = masksSubtask
        self.run_after(masksSubtask)
        self.startYear = startYear
        self.endYear = endYear
        self.densityBins = densityBins
        self.fluxVariables = fluxVariables
        self.startDate = f'{self.startYear:04d}-01-01_00:00:00'
        self.endDate = f'{self.endYear:04d}-12-31_23:59:59'
        self.regionGroups = regionGroups
        self.regionNames = regionNames

    def setup_and_check(self):
        """
        Perform steps to set up the analysis and check for errors in the setup.

        Raises
        ------
        IOError
            If an input file is not present
        """
        super(ComputeRegionWmtSubtask, self).setup_and_check()

        # get a list of timeMonthlyAvg output files from the streams file,
        # reading only those that are between the start and end dates
        self.streamName = 'timeSeriesStatsMonthlyOutput'
        self.inputFiles = self.historyStreams.readpath(
            self.streamName, startDate=self.startDate, endDate=self.endDate,
            calendar=self.calendar)

        if len(self.inputFiles) == 0:
            raise IOError('No files were found in stream {} between {} and '
                          '{}.'.format(self.streamName, self.startDate,
                                       self.endDate))

        prefix = 'timeMonthly_avg'
        self.variableList = ['Time', # 'xtime_startMonthly', 'xtime_endMonthly',
                             f'{prefix}_activeTracers_temperature',
                             f'{prefix}_activeTracers_salinity']
        for variable in self.fluxVariables:
            self.variableList.append(f'{prefix}_{variable}')

    def run_task(self):

        iselValues = {'Time': 0,
                      'nVertLevels': 0}
        surfacePressure = 0. # dbar

        nBins = len(self.densityBins) - 1

        # Loop over time by loading inputFiles
        ds_out = xarray.Dataset()
        for fileCount, inputFile in enumerate(self.inputFiles):

            print(inputFile)
            ds = open_mpas_dataset(fileName=inputFile,
                                   calendar=self.calendar,
                                   variableList=self.variableList,
                                   startDate=self.startDate,
                                   endDate=self.endDate)
            year = self.startYear
            ds = ds.isel(iselValues)

            for regionGroup in self.regionGroups:
                regionGroupSuffix = regionGroup.replace(' ', '_')
                output_filename = _get_regional_wmt_file_name(self.config, startYear=year, endYear=year)

                maskFileName = self.masksSubtask.maskFileName
                print(f'open {maskFileName}')
                with xarray.open_dataset(maskFileName) as dsRegionMask:
                    maskRegionNames = decode_strings(dsRegionMask.regionNames)
                    regionIndices = []
                    for regionName in self.regionNames:
                        for index, otherName in enumerate(maskRegionNames):
                            if regionName == otherName:
                                regionIndices.append(index)
                                break

                # select only those regions we want to plot
                dsRegionMask = dsRegionMask.isel(nRegions=regionIndices)

                nRegions = dsRegionMask.sizes['nRegions']
                dsRegionGroup = xarray.Dataset()

                datasets = []
                for regionIndex in range(nRegions):
                    regionName = self.regionNames[regionIndex]
                    self.logger.info(f'    region: {regionName}')
                    regionNameSuffix = regionName.replace(' ', '_')
                    # Downselect cells to those in the region
                    dsMask = dsRegionMask.isel(nRegions=regionIndex)
                    cellMask = dsMask.regionCellMasks == 1
                    # TODO consider open ocean masking here following time_series_ocean_regions
                    ds = ds.where(cellMask, drop=True)
                    print(ds.sizes)
                    nCells = ds.sizes['nCells']

                    ds_new = xarray.Dataset(
                        data_vars=dict(
                            regionNames=(["nRegions"], regionName),
                            densityMask=(["Time", "nRegions", "nBins", "nCells"],
                                         numpy.zeros((1, 1, nBins, nCells)))
                        ),
                        coords=dict(
                            density_min=("nBins", self.densityBins[:-1]),
                            density_max=("nBins", self.densityBins[1:]),
                            Time=("Time", ds.Time.values)
                            #xtime_startMonthly=("Time", ds.xtime_startMonthly),
                            #xtime_endMonthly=("Time", ds.xtime_endMonthly)
                        )
                    )
                    # Compute density from T,S
                    prefix = 'timeMonthly_avg'
                    temperature = ds[f'{prefix}_activeTracers_temperature']
                    salinity = ds[f'{prefix}_activeTracers_salinity']
                    density = rho(salinity, temperature, surface_pressure)

                    # Loop over density bins
                    for iBin in range(nBins):
        
                        density_min = densityBins[iBin]
                        density_max = densityBins[iBin + 1]
                        # Compute density bin mask
                        density_mask = xarray.logical_and(density >= density_min,
                                                      density < density_max) 

                        ds_new['densityMask'][0, 0, :, :] = density_mask

                    # Loop over flux variables
                    for variable in self.fluxVariable:
                        # Loop over density bins
                        for iBin in range(nBins):
                            # Apply density bin mask to flux variable
                            flux_masked[iBin, :] = ds[f'timeMonthly_avg_{variable}'].values * mask
                            # Make sure this is saved to the right region

                        ds_new['timeMonthly_avg_{variable}'][0, 0, :, :] = flux_masked

                    dsRegion = xarray.concat([dsRegion, ds_new], dim='Time')
                    dsRegion.coords['regionNames'] = dsRegion['regionNames']
                    print('dsRegion sizes = ',dsRegion.sizes)

                datasets.append(dsRegion)

            dsRegionGroup = xarray.concat(objs=datasets, dim='nRegions')
            print('dsRegionGroup sizes = ',dsRegionGroup.sizes)

            # save one file for each region group which gets added to every time step
            write_netcdf_with_fill(dsRegionGroup, output_filename)

    
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
                 sectionName, mpasMasksSubtask, fluxVariables):

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

        mpasMasksSubtask : ``ComputeRegionMasksSubtask``
            A task for creating mask MPAS files for each region to plot, used
            to get the mask file name
        """

        # first, call the constructor from the base class (AnalysisTask)
        regionGroupSuffix = regionGroup.replace(' ', '_')
        regionNameSuffix = regionGroup.replace(' ', '_')
        filePrefix = f'wmt_{regionGroupSuffix}_{regionNameSuffix}'
        super(PlotRegionWmtSubtask, self).__init__(
            config=parentTask.config,
            taskName=parentTask.taskName,
            componentName=parentTask.componentName,
            tags=parentTask.tags,
            subtaskName=f'plot_{filePrefix}')

        self.regionGroup = regionGroup
        self.regionName = regionName
        self.sectionName = sectionName
        self.controlConfig = controlConfig
        self.mpasMasksSubtask = mpasMasksSubtask
        self.filePrefix = filePrefix
        self.fluxVariables = fluxVariables

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
        self.xmlFileNames.append(
            f'{self.plotsDirectory}/{self.filePrefix}.xml')

        config = self.config
        startYear = config.getint(self.taskName, 'startYear')
        endYear = config.getint(self.taskName, 'endYear')
        self.inputFileNames = []
        for year in range(startYear, endYear + 1):
            filename = _get_regional_wmt_file_name(config, startYear=year, endYear=year)
            if os.path.exists(filename):
                self.inputFileNames.append()

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

        # ds_bins = filenames[0]
        # ds_values = []

        # create a single dataset corresponding to the region and whole time window considered
        ds_alltime = xarray.Dataset()
        for filename in self.inputFileNames:
            with xarray.open_dataset(filename) as ds:
                ds_region = ds.isel(regionNames=self.regionName)
                ds_alltime = xarray.concat([ds_alltime, ds_region], dim='Time')
        # If we wanted to select seasons we could do so here
        ds_out = ds_alltime.mean(dim='Time')
        output_file_name = f'wmt_{self.regionGroup}_{self.regionName}.nc'
        write_netcdf_with_fill(ds_out, output_file_name)

        # average over nCellsRegion to get ds_values corresponding to each bin
        for rho_bin in bins:
            ds_rho = ds_out # mean over rho bin
            for var in self.fluxVariables:
                ds_values.append(ds_rho.var)

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

def _get_regional_wmt_file_name(config, startYear, endYear,
    season='ANN'):

    monthValues = sorted(constants.monthDictionary[season])
    startMonth = monthValues[0]
    endMonth = monthValues[-1]

    baseDirectory = build_config_full_path(config, 'output', 'wmtSubdirectory')
    suffix = '{:04d}{:02d}_{:04d}{:02d}_climo'.format(
        startYear, startMonth, endYear, endMonth)

    if season in constants.abrevMonthNames:
        season = '{:02d}'.format(monthValues[0])
    fileName = f'{baseDirectory}/wmt_{season}_{suffix}.nc'
    return fileName
