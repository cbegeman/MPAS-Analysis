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
    def __init__(self, parentTask, regionGroup, regionNames, startYear, endYear,
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
        self.regionGroup = regionGroup
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
        surface_pressure = 0. # dbar

        nBins = len(self.densityBins) - 1
        nTime = len(self.inputFiles)

        # Loop over time by loading inputFiles
        ds_out = xarray.Dataset()
        year = self.startYear
        regionGroupSuffix = self.regionGroup.replace(' ', '_')

        maskFileName = self.masksSubtask.maskFileName
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
        print(f'Create dataset for {self.regionGroup}')
        out_datasets = []
        out_filenames = []
        for regionIndex, regionName in enumerate(self.regionNames):
            dsFull = open_mpas_dataset(fileName=self.inputFiles[0],
                                       calendar=self.calendar,
                                       variableList=self.variableList,
                                       startDate=self.startDate,
                                       endDate=self.endDate)
            out_filenames.append(
                _get_regional_wmt_file_name(
                    self.config, startYear=year, endYear=year,
                    regionGroup=self.regionGroup, regionName=regionName))
            # Downselect cells to those in the region
            dsMask = dsRegionMask.isel(nRegions=regionIndex)
            cellMask = dsMask.regionCellMasks == 1
            print(f'dsMask sizes: {dsMask.sizes}')
            # TODO consider open ocean masking here following time_series_ocean_regions
            ds = dsFull.where(cellMask, drop=True)
            nCells = ds.sizes['nCells']
            print('ds sizes = ',ds.sizes)
            print('ds keys = ',ds.keys())

            dsRegion = xarray.Dataset(
                data_vars=dict(
                    #xtime=(["Time"], numpy.zeros((1))), # ds.Time.values),
                    densityMask=(["Time", "nRegions", "nBins", "nCells"],
                                 numpy.zeros((nTime, 1, nBins, nCells), dtype=bool))
                ),
                coords=dict(
                    density_min=("nBins", self.densityBins[:-1]),
                    density_max=("nBins", self.densityBins[1:]),
                    regionNames=("nRegions", [regionName])
                    #Time=("Time", ds.Time.values)
                    #xtime_startMonthly=("Time", ds.xtime_startMonthly),
                    #xtime_endMonthly=("Time", ds.xtime_endMonthly)
                )
            )
            for variable in self.fluxVariables:
                dsRegion[variable] = \
                    (("Time", "nRegions", "nCells"), numpy.zeros((nTime, 1, nCells)))
            out_datasets.append(dsRegion)

        for fileCount, inputFile in enumerate(self.inputFiles):

            print(f'load ds: {inputFile}')
            dsFull = open_mpas_dataset(fileName=inputFile,
                                       calendar=self.calendar,
                                       variableList=self.variableList,
                                       startDate=self.startDate,
                                       endDate=self.endDate)
            dsFull = dsFull.isel(iselValues)
            print('dsFull isel sizes = ',dsFull.sizes)

            for regionIndex in range(nRegions):
                regionName = self.regionNames[regionIndex]
                self.logger.info(f'    region: {regionName}')
                print(f'    region: {regionName}')

                dsRegion = out_datasets[regionIndex]
                print('dsRegion sizes = ',dsRegion.sizes)

                # Downselect cells to those in the region
                dsMask = dsRegionMask.isel(nRegions=regionIndex)
                cellMask = dsMask.regionCellMasks == 1
                print(f'dsMask sizes: {dsMask.sizes}')
                # TODO consider open ocean masking here following time_series_ocean_regions
                ds = dsFull.where(cellMask, drop=True)
                print('dsFull sizes = ',dsFull.sizes)
                # Compute density from T,S
                prefix = 'timeMonthly_avg'
                temperature = ds[f'{prefix}_activeTracers_temperature']
                salinity = ds[f'{prefix}_activeTracers_salinity']
                density = rho(salinity, temperature, surface_pressure)
                #print(ds.Time)
                #ds_new['xtime'] = ds.Time

                # Loop over density bins
                for iBin in range(nBins):
        
                    density_min = self.densityBins[iBin]
                    density_max = self.densityBins[iBin + 1]
                    # Compute density bin mask
                    density_mask = numpy.logical_and(density.values >= density_min,
                                                     density.values < density_max) 
                    print(f'density_mask size = {numpy.shape(density_mask)}')
                    dsRegion['densityMask'][fileCount, 0, iBin, :] = density_mask

                # Loop over flux variables
                for variable in self.fluxVariables:
                    flux = dsRegion[variable].values
                    print(f'flux size = {numpy.shape(flux)}')
                    flux[fileCount, 0, :] = ds[f'{prefix}_{variable}']
                    dsRegion[variable].values = flux

        for regionIndex in range(nRegions):
            print(f'write file {out_filenames[regionIndex]}')
            write_netcdf_with_fill(out_datasets[regionIndex], out_filenames[regionIndex])

    
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
        regionNameSuffix = regionName.replace(' ', '_')
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
        self.startYear = startYear
        self.endYear = endYear
        for year in range(startYear, endYear + 1):
            filename = _get_regional_wmt_file_name(config, startYear=year, endYear=year,
                regionGroup=self.regionGroup, regionName=self.regionName)
            print(f'Search for input file {filename}')
            if os.path.exists(filename):
                self.inputFileNames.append(filename)

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

        # create a single dataset corresponding to the region and whole time window considered
        for fileCount, filename in enumerate(self.inputFileNames):
            print(f'load file to plot: {filename}')
            with xarray.open_dataset(filename) as ds_region:
                ds_region = ds_region.isel(nRegions=0)
                print(ds_region.sizes)
                if fileCount == 0:
                    ds_alltime = ds_region.copy()
                else:
                    ds_alltime = xarray.concat([ds_alltime, ds_region], dim='Time')
        print(ds_alltime.sizes)

        # average over nCellsRegion to get ds_values corresponding to each bin
        ds_bins = numpy.concat((ds_alltime.density_min.values,
                                [ds_alltime.density_max.values[-1]]))
        print(f'ds_bins shape {numpy.shape(ds_bins)}')
        for var in self.fluxVariables:
            flux_masked = numpy.zeros((ds_alltime.sizes['Time'],
                                       ds_alltime.sizes['nBins']))
            for iBin in range(ds_alltime.sizes['nBins']):
                for iTime in range(ds_alltime.sizes['Time']):
                    flux = ds_alltime[var].isel(Time=iTime).values
                    mask = ds_alltime['densityMask'].isel(Time=iTime, nBins=iBin).values
                    # Apply density bin mask to flux variable
                    if numpy.sum(mask==1) == 0:
                        flux_masked[iTime, iBin] = 0.
                    else:
                        flux_masked[iTime, iBin] = numpy.nanmean(flux[mask])
            ds_alltime[f'{var}_binned'] = (("Time", "nBins"), flux_masked)

        # by taking mean over nCells, we are actually taking mean over one rho bin 
        # If we wanted to select seasons we could do so here rather than averaging over all time
        ds = ds_alltime.mean(dim='Time')

        output_file_name = _get_regional_wmt_file_name(
            config, startYear=self.startYear, endYear=self.endYear,
            regionGroup=self.regionGroup, regionName=self.regionName,
            time_averaged=True)
        print(f'write averaged file to {output_file_name}')
        write_netcdf_with_fill(ds, output_file_name)

        wmt_yearly_plot(config, ds, mode='cumulative')

        #file_prefix = f'{self.filePrefix}_' \
        #               'cumulative_surface_flux_' \
        #               f'{self.regionName}_wmt'
        #caption = f'Climatological surface flux WMT for ' \
        #          f'{self.regionName.replace("_", " ")}'
        #out_filename = f'{self.plotsDirectory}/{file_prefix}.png'
        #savefig(out_filename, config)

        #write_image_xml(
        #    config=config,
        #    filePrefix=file_prefix,
        #    componentName='Ocean',
        #    componentSubdirectory='ocean',
        #    galleryGroup=f'{self.regionGroup} WMT',
        #    groupLink=f'wmtCumulative',
        #    gallery='wmt',
        #    thumbnailDescription=f'{self.regionName.replace("_", " ")} ',
        #    imageDescription=caption,
        #    imageCaption=caption)

def _get_regional_wmt_file_name(config, startYear, endYear,
    regionGroup, regionName, season='ANN', time_averaged=False):

    regionGroupSuffix = regionGroup.replace(' ', '_')
    regionNameSuffix = regionName.replace(' ', '_')
    monthValues = sorted(constants.monthDictionary[season])
    startMonth = monthValues[0]
    endMonth = monthValues[-1]

    baseDirectory = build_config_full_path(config, 'output', 'wmtSubdirectory')
    suffix = '{:04d}{:02d}_{:04d}{:02d}_climo'.format(
        startYear, startMonth, endYear, endMonth)

    if season in constants.abrevMonthNames:
        season = '{:02d}'.format(monthValues[0])
    if time_averaged:
        prefix = 'wmt_tavg'
    else:
        prefix = 'wmt'
    fileName = f'{baseDirectory}/{prefix}_{regionGroupSuffix}_{regionNameSuffix}_{season}_{suffix}.nc'
    return fileName
