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
# https://raw.githubusercontent.com/MPAS-Dev/MPAS-Analysis/master/LICENSE
#
import os
import xarray
import numpy
import matplotlib.pyplot as plt

from mpas_analysis.shared import AnalysisTask

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf
from mpas_analysis.shared.io.utility import build_config_full_path, \
    build_obs_path, make_directories, decode_strings
from mpas_analysis.shared.climatology import compute_climatology, \
    get_unmasked_mpas_climatology_file_name

from mpas_analysis.shared.constants import constants
from mpas_analysis.shared.plot import histogram_analysis_plot, savefig
from mpas_analysis.shared.html import write_image_xml


class OceanHistogram(AnalysisTask):
    """
    Plots a histogram of a 2-d ocean variable.

    """

    def __init__(self, config, mpasClimatologyTask, regionMasksTask,
                 controlConfig=None):

        """
        Construct the analysis task.

        Parameters
        ----------
        config : mpas_tools.config.MpasConfigParser
            Configuration options

        mpasClimatologyTask : ``MpasClimatologyTask``
            The task that produced the climatology to be remapped and plotted

        regionMasksTask : ``ComputeRegionMasks``
            A task for computing region masks

        controlConfig : mpas_tools.config.MpasConfigParser
            Configuration options for a control run (if any)
        """

        # first, call the constructor from the base class (AnalysisTask)
        super().__init__(
            config=config,
            taskName='oceanHistogram',
            componentName='ocean',
            tags=['climatology', 'regions', 'histogram', 'publicObs'])

        self.run_after(mpasClimatologyTask)
        self.mpasClimatologyTask = mpasClimatologyTask

        self.controlConfig = controlConfig
        mainRunName = config.get('runs', 'mainRunName')

        self.regionGroups = config.getexpression(self.taskName, 'regionGroups')
        self.regionNames = config.getexpression(self.taskName, 'regionNames')
        self.seasons = config.getexpression(self.taskName, 'seasons')
        self.variableList = config.getexpression(self.taskName, 'variableList')
        if config.has_option(self.taskName, 'weightList'):
            self.weightList = config.getexpression(self.taskName, 'weightList')
        else:
            self.weightList = None

        self.filePrefix = f'histogram_{mainRunName}'

        baseDirectory = build_config_full_path(
            config, 'output', 'histogramSubdirectory')
        if not os.path.exists(baseDirectory):
            make_directories(baseDirectory)

        self.obsList = config.getexpression(self.taskName, 'obsList')
        obsDicts = {
            'AVISO': {
                'suffix': 'AVISO',
                'gridName': 'Global_1.0x1.0degree',
                'gridFileName': 'SSH/zos_AVISO_L4_199210-201012_20180710.nc',
                'lonVar': 'lon',
                'latVar': 'lat',
                'sshVar': 'zos',
                'pressureAdjustedSSHVar': 'zos'}}

        for regionGroup in self.regionGroups:
            groupObsDicts = {}
            mpasMasksSubtask = regionMasksTask.add_mask_subtask(
                regionGroup=regionGroup)
            regionNames = mpasMasksSubtask.expand_region_names(
                self.regionNames)

            # Add mask subtasks for observations and prep groupObsDicts
            # groupObsDicts is a subsetted version of localObsDicts with an
            # additional attribute for the maskTask
            for obsName in self.obsList:
                localObsDict = dict(obsDicts[obsName])
                obsFileName = build_obs_path(
                    config, component=self.componentName,
                    relativePath=localObsDict['gridFileName'])
                obsMasksSubtask = regionMasksTask.add_mask_subtask(
                    regionGroup, obsFileName=obsFileName,
                    lonVar=localObsDict['lonVar'],
                    latVar=localObsDict['latVar'],
                    meshName=localObsDict['gridName'])
                localObsDict['maskTask'] = obsMasksSubtask
                groupObsDicts[obsName] = localObsDict

            for regionName in regionNames:
                sectionName = None

                # Compute weights for histogram
                if self.weightList is not None:
                    computeWeightsSubtask = ComputeHistogramWeightsSubtask(
                        self, regionName, mpasMasksSubtask, self.filePrefix,
                        self.variableList, self.weightList)
                    self.add_subtask(computeWeightsSubtask)

                for season in self.seasons:

                    # Generate histogram plots
                    plotRegionSubtask = PlotRegionHistogramSubtask(
                        self, regionGroup, regionName, controlConfig,
                        sectionName, self.filePrefix, mpasClimatologyTask,
                        mpasMasksSubtask, obsMasksSubtask, groupObsDicts,
                        self.variableList, self.weightList, season)
                    plotRegionSubtask.run_after(mpasMasksSubtask)
                    plotRegionSubtask.run_after(obsMasksSubtask)
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

        # Add variables and seasons to climatology task
        variableList = []
        for var in self.variableList:
            variableList.append(f'timeMonthly_avg_{var}')

        self.mpasClimatologyTask.add_variables(variableList=variableList,
                                               seasons=self.seasons)

        if self.weightList is not None:
            if len(self.weightList) != len(self.variableList):
                raise ValueError('Histogram weightList is not the same '
                                 'length as variableList')
        if len(self.obsList) > 1:
            raise ValueError('Histogram analysis does not currently support'
                             'more than one observational product')


class ComputeHistogramWeightsSubtask(AnalysisTask):
    """
    Fetches weight variables from MPAS output files for each variable in
    variableList.

    """
    def __init__(self, parentTask, regionName, mpasMasksSubtask, fullSuffix,
                 variableList, weightList):
        """
        Initialize weights task

        Parameters
        ----------
        parentTask :  ``AnalysisTask``
            The parent task, used to get the ``taskName``, ``config`` and
            ``componentName``

        regionName : str
            Name of the region to plot

        mpasMasksSubtask : ``ComputeRegionMasksSubtask``
            A task for creating mask MPAS files for each region to plot, used
            to get the mask file name

        obsDicts : dict of dicts
            Information on the observations to compare agains

        fullSuffix : str
            The regionGroup and regionName combined and modified to be
            appropriate as a task or file suffix

        variableList: list of str
            List of variables which will be weighted

        weightList: list of str
            List of variables by which to weight the variables in
            variableList, of the same length as variableList

        """

        super(ComputeHistogramWeightsSubtask, self).__init__(
            config=parentTask.config,
            taskName=parentTask.taskName,
            componentName=parentTask.componentName,
            tags=parentTask.tags,
            subtaskName=f'weights{fullSuffix}_{regionName}')

        self.mpasMasksSubtask = mpasMasksSubtask
        self.regionName = regionName
        self.filePrefix = fullSuffix
        self.variableList = variableList
        self.weightList = weightList

    def run_task(self):
        """
        Apply the region mask to each weight variable and save in a file common
        to that region.
        """

        config = self.config

        # Get cell mask for the region
        region_mask_filename = self.mpasMasksSubtask.maskFileName
        ds_region_mask = xarray.open_dataset(region_mask_filename)
        mask_region_names = decode_strings(ds_region_mask.regionNames)
        region_index = mask_region_names.index(self.regionName)
        ds_mask = dsRegionMask.isel(nRegions=region_index)
        cell_mask = dsMask.regionCellMasks == 1

        # Open the restart file, which contains unmasked weight variables
        restart_filename = self.runStreams.readpath('restart')[0]
        ds_restart = xarray.open_dataset(restart_filename)
        ds_restart = ds_restart.isel(Time=0)

        # Save the cell mask only for the region in its own file, which may be
        # referenced by future analysis (i.e., as a control run)
        new_region_mask_filename = \
            f'{baseDirectory}/{self.filePrefix}_{self.regionName}_mask.nc'
        write_netcdf(dsMask, new_region_mask_filename)

        ds_weights = xarray.Dataset()
        # Fetch the weight variables and mask them for each region
        for index, var in enumerate(self.variableList):
            weight_var_name = self.weightList[index]
            if weight_var_name in dsRestart.keys():
                var_name = f'timeMonthly_avg_{var}'
                ds_weights[f'{varname}_weight'] = \
                    ds_restart[weight_var_name].where(cell_mask, drop=True)
            else:
                self.logger.warn(f'Weight variable {weight_var_name} is not in '
                                 f'the restart file, skipping')

        base_directory = build_config_full_path(
            config, 'output', 'histogramSubdirectory')
        weights_filename = \
            f'{baseDirectory}/{self.filePrefix}_{self.regionName}_weights.nc'
        write_netcdf(ds_weights, weights_filename)


class PlotRegionHistogramSubtask(AnalysisTask):
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

    mpasClimatologyTask : ``MpasClimatologyTask``
        The task that produced the climatology to be remapped and plotted

    mpasMasksSubtask : ``ComputeRegionMasksSubtask``
        A task for creating mask MPAS files for each region to plot, used
        to get the mask file name

    obsDicts : dict of dicts
        Information on the observations to compare against

    variableList: list of str
        list of variables to plot

    season : str
        The season to compute the climatology for
    """

    def __init__(self, parentTask, regionGroup, regionName, controlConfig,
                 sectionName, fullSuffix, mpasClimatologyTask,
                 mpasMasksSubtask, obsMasksSubtask, obsDicts, variableList,
                 weightList, season):

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

        mpasClimatologyTask : ``MpasClimatologyTask``
            The task that produced the climatology to be remapped and plotted

        mpasMasksSubtask : ``ComputeRegionMasksSubtask``
            A task for creating mask MPAS files for each region to plot, used
            to get the mask file name

        obsDicts : dict of dicts
            Information on the observations to compare agains

        season : str
            The season to comput the climatogy for
        """

        # first, call the constructor from the base class (AnalysisTask)
        super(PlotRegionHistogramSubtask, self).__init__(
            config=parentTask.config,
            taskName=parentTask.taskName,
            componentName=parentTask.componentName,
            tags=parentTask.tags,
            subtaskName=f'plot{fullSuffix}_{regionName}_{season}')

        self.run_after(mpasClimatologyTask)
        self.regionGroup = regionGroup
        self.regionName = regionName
        self.sectionName = sectionName
        self.controlConfig = controlConfig
        self.mpasClimatologyTask = mpasClimatologyTask
        self.mpasMasksSubtask = mpasMasksSubtask
        self.obsMasksSubtask = obsMasksSubtask
        self.obsDicts = obsDicts
        self.variableList = variableList
        self.weightList = weightList
        self.season = season
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
        super(PlotRegionHistogramSubtask, self).setup_and_check()

        self.xmlFileNames = []
        for var in self.variableList:
            self.xmlFileNames.append(
                f'{self.plotsDirectory}/{self.filePrefix}_{var}_'
                f'{self.regionName}_{self.season}.xml')

    def run_task(self):
        """
        Plots histograms of properties in an ocean region.
        """

        self.logger.info(f"\nPlotting {self.season} histograms for "
                         f"{self.regionName}")

        config = self.config
        sectionName = self.sectionName

        calendar = self.calendar

        mainRunName = config.get('runs', 'mainRunName')

        baseDirectory = build_config_full_path(
            config, 'output', 'histogramSubdirectory')

        regionMaskFileName = self.mpasMasksSubtask.maskFileName

        dsRegionMask = xarray.open_dataset(regionMaskFileName)

        maskRegionNames = decode_strings(dsRegionMask.regionNames)
        regionIndex = maskRegionNames.index(self.regionName)

        dsMask = dsRegionMask.isel(nRegions=regionIndex)
        cellMask = dsMask.regionCellMasks == 1

        inFileName = get_unmasked_mpas_climatology_file_name(
            config, self.season, self.componentName, op='avg')

        if len(self.obsDicts) > 0:
            obsRegionMaskFileName = self.obsMasksSubtask.maskFileName
            dsObsRegionMask = xarray.open_dataset(obsRegionMaskFileName)
            maskRegionNames = decode_strings(dsRegionMask.regionNames)
            regionIndex = maskRegionNames.index(self.regionName)

            dsObsMask = dsObsRegionMask.isel(nRegions=regionIndex)
            obsCellMask = dsObsMask.regionMasks == 1

        ds = xarray.open_dataset(inFileName)
        ds = ds.where(cellMask, drop=True)

        baseDirectory = build_config_full_path(
            config, 'output', 'histogramSubdirectory')
        if self.weightList is not None:
            weightsFileName = \
                f'{baseDirectory}/{self.filePrefix}_{self.regionName}_' \
                'weights.nc'
            dsWeights = xarray.open_dataset(weightsFileName)

        if self.controlConfig is not None:
            controlRunName = self.controlConfig.get('runs', 'mainRunName')
            controlFileName = get_unmasked_mpas_climatology_file_name(
                self.controlConfig, self.season, self.componentName, op='avg')
            dsControl = xarray.open_dataset(controlFileName)
            baseDirectory = build_config_full_path(
                self.controlConfig, 'output', 'histogramSubdirectory')
            controlRegionMaskFileName = f'{baseDirectory}/histogram_' \
                f'{controlRunName}_{self.regionName}_mask.nc'
            dsControlRegionMasks = xarray.open_dataset(
                controlRegionMaskFileName)
            controlCellMask = dsControlRegionMasks.regionCellMasks == 1
            if self.weightList is not None:
                controlWeightsFileName = \
                    f'{baseDirectory}/histogram_{controlRunName}_' \
                    f'{self.regionName}_weights.nc'
                dsControlWeights = xarray.open_dataset(controlWeightsFileName)

        if config.has_option(self.taskName, 'lineColors'):
            lineColors = [config.get(self.taskName, 'mainColor')]
        else:
            lineColors = None
        if config.has_option(self.taskName, 'obsColor'):
            obsColor = config.get_expression(self.taskName, 'obsColor')
            if lineColors is None:
                lineColors = ['b']
        else:
            if lineColors is not None:
                obsColor = 'k'

        if config.has_option(self.taskName, 'lineWidths'):
            lineWidths = [config.get(self.taskName, 'lineWidths')]
        else:
            lineWidths = None

        title = mainRunName
        if config.has_option(self.taskName, 'titleFontSize'):
            titleFontSize = config.getint(self.taskName,
                                          'titleFontSize')
        else:
            titleFontSize = None
        if config.has_option(self.taskName, 'titleFontSize'):
            axisFontSize = config.getint(self.taskName,
                                         'axisFontSize')
        else:
            axisFontSize = None

        if config.has_option(self.taskName, 'defaultFontSize'):
            defaultFontSize = config.getint(self.taskName,
                                            'defaultFontSize')
        else:
            defaultFontSize = None
        if config.has_option(self.taskName, 'bins'):
            bins = config.getint(self.taskName, 'bins')
        else:
            bins = None

        yLabel = 'normalized Probability Density Function'

        for index, var in enumerate(self.variableList):

            fields = []
            weights = []
            legendText = []

            varname = f'timeMonthly_avg_{var}'

            caption = f'Normalized probability density function for {var} ' \
                      f'climatologies in {self.regionName.replace("_", " ")}'

            # Note: consider modifying this for more professional headings
            varTitle = var

            fields.append(ds[varname])
            if self.weightList is not None:
                if f'{varname}_weight' in dsWeights.keys():
                    weights.append(dsWeights[f'{varname}_weight'].values)
                    caption = f'{caption} weighted by {self.weightList[index]}'
                else:
                    weights.append(None)
            else:
                weights.append(None)

            legendText.append(mainRunName)

            xLabel = f"{ds[varname].attrs['long_name']} " \
                     f"({ds[varname].attrs['units']})"

            for obsName in self.obsDicts:
                localObsDict = dict(self.obsDicts[obsName])
                obsFileName = build_obs_path(
                    config, component=self.componentName,
                    relativePath=localObsDict['gridFileName'])
                if f'{var}Var' not in localObsDict.keys():
                    self.logger.warn(
                        f'{var}Var is not present in {obsName}, skipping '
                        f'{obsName}')
                    continue
                varnameObs = localObsDict[f'{var}Var']
                dsObs = xarray.open_dataset(obsFileName)
                dsObs = dsObs.where(obsCellMask, drop=True)
                fields.append(dsObs[varnameObs])
                legendText.append(obsName)
                if lineColors is not None:
                    lineColors.append(obsColor)
                if lineWidths is not None:
                    lineWidths.append([lineWidths[0]])
                weights.append(None)
            if self.controlConfig is not None:
                fields.append(dsControl[varname].where(controlCellMask,
                                                       drop=True))
                controlRunName = self.controlConfig.get('runs', 'mainRunName')
                legendText.append('Control')
                title = f'{title} vs. {controlRunName}'
                if lineColors is not None:
                    lineColors.append(obsColor)
                if lineWidths is not None:
                    lineWidths.append([lineWidths[0]])
                weights.append(dsControlWeights[f'{varname}_weight'].values)
            histogram_analysis_plot(config, fields, calendar=calendar,
                                    title=title, xlabel=xLabel, ylabel=yLabel,
                                    bins=bins, weights=weights,
                                    lineColors=lineColors,
                                    lineWidths=lineWidths,
                                    legendText=legendText,
                                    titleFontSize=titleFontSize,
                                    defaultFontSize=defaultFontSize)

            outFileName = f'{self.plotsDirectory}/{self.filePrefix}_{var}_' \
                          f'{self.regionName}_{self.season}.png'
            savefig(outFileName, config)

            write_image_xml(
                config=config,
                filePrefix=f'{self.filePrefix}_{var}_{self.regionName}_'
                           f'{self.season}',
                componentName='Ocean',
                componentSubdirectory='ocean',
                galleryGroup=f'{self.regionGroup} Histograms',
                groupLink=f'histogram{var}',
                gallery=varTitle,
                thumbnailDescription=f'{self.regionName.replace("_", " ")} '
                                     f'{self.season}',
                imageDescription=caption,
                imageCaption=caption)
