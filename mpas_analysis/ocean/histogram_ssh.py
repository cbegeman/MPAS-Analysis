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
import numpy
import matplotlib.pyplot as plt

from mpas_analysis.shared import AnalysisTask

from mpas_analysis.shared.io import open_mpas_dataset
from mpas_analysis.shared.io.utility import build_config_full_path
from mpas_analysis.shared.climatology import compute_climatology#, \
#    get_unmasked_mpas_climatology_file_name, \
#    get_masked_mpas_climatology_file_name

from mpas_analysis.shared.constants import constants
from mpas_analysis.shared.plot import histogram_analysis_plot, savefig
from mpas_analysis.shared.html import write_image_xml

class HistogramSSH(AnalysisTask):
    """
    Plots a histogram of the global mean sea surface height.

    Attributes
    ----------
    variableDict : dict
        A dictionary of variables from the time series stats monthly output
        (keys), together with shorter, more convenient names (values)

    histogramFileName : str
        The name of the file where the ssh histogram is stored

    controlConfig : mpas_tools.config.MpasConfigParser
        Configuration options for a control run (if one is provided)

    filePrefix : str
        The basename (without extension) of the PNG and XML files to write out
    """
    # Authors
    # -------
    # Xylar Asay-Davis

    def __init__(self, config, mpasClimatologyTask, regionMasksTask, controlConfig=None):

        """
        Construct the analysis task.

        Parameters
        ----------
        config : mpas_tools.config.MpasConfigParser
            Configuration options

        mpasHistogram: ``MpasHistogramTask``
            The task that extracts the time series from MPAS monthly output

        mpasClimatologyTask : ``MpasClimatologyTask``
            The task that produced the climatology to be remapped and plotted

        regionMasksTask : ``ComputeRegionMasks``
            A task for computing region masks

        controlConfig : mpas_tools.config.MpasConfigParser
            Configuration options for a control run (if any)
        """
        # Authors
        # -------
        # Xylar Asay-Davis

        # first, call the constructor from the base class (AnalysisTask)
        super().__init__(
            config=config,
            taskName='histogramSSH',
            componentName='ocean',
            tags=['climatology', 'regions', 'histogram', 'ssh', 'publicObs'])

        self.run_after(mpasClimatologyTask)
        self.mpasClimatologyTask = mpasClimatologyTask

        #self.histogramFileName = ''
        self.controlConfig = controlConfig
        self.filePrefix = None

        self.variableDict = {}
        for var in ['ssh']:
            key = 'timeMonthly_avg_{}'.format(var)
            self.variableDict[key] = var

    def setup_and_check(self):
        """
        Perform steps to set up the analysis and check for errors in the setup.

        Raises
        ------
        OSError
            If files are not present
        """
        # Authors
        # -------
        # Xylar Asay-Davis

        # first, call setup_and_check from the base class (AnalysisTask),
        # which will perform some common setup, including storing:
        #   self.inDirectory, self.plotsDirectory, self.namelist, self.streams
        #   self.calendar
        super().setup_and_check()

        config = self.config

        mainRunName = config.get('runs', 'mainRunName')
        self.startYear = self.config.getint(self.taskName, 'startYear')
        self.endYear = self.config.getint(self.taskName, 'endYear')
        regionGroups = config.getexpression(self.taskName, 'regionGroups')

        self.seasons = config.getexpression(self.taskName, 'seasons')


        self.xmlFileNames = []

        self.filePrefix = 'ssh_histogram_{}'.format(mainRunName)
        self.xmlFileNames.append('{}/{}.xml'.format(self.plotsDirectory,
                                                    self.filePrefix))

    def run_task(self):
        """
        Performs histogram analysis of the output of sea-surface height
        (SSH).
        """
        # Authors
        # -------
        # Carolyn Begeman, Adrian Turner, Xylar Asay-Davis

        self.logger.info("\nPlotting histogram of SSH...")

        config = self.config
        calendar = self.calendar
        seasons = self.seasons

        startYear = self.startYear
        endYear = self.endYear
        #TODO determine whether this is needed
        #startDate = '{:04d}-01-01_00:00:00'.format(self.startYear)
        #endDate = '{:04d}-12-31_23:59:59'.format(self.endYear)

        mainRunName = config.get('runs', 'mainRunName')

        variableList = ['ssh']

        baseDirectory = build_config_full_path(
            config, 'output', 'histogramSubdirectory')
        print(f'baseDirectory={baseDirectory}')
        print(f'plotsDirectory={self.plotsDirectory}')

        # the variable mpasFieldName will be added to mpasClimatologyTask
        # along with the seasons.

        for season in seasons:
            outputFileName = \
                '{}/{}_{}_{:04d}-{:04d}.nc'.format(
                    baseDirectory, 'histogramSSH', season,
                    startYear, endYear)
            outFileName = f'{baseDirectory}/{self.filePrefix}_{season}_{startYear}-{endYear}'
            #outFileName = f'{self.plotsDirectory}/{self.filePrefix}_{season}_{startYear}-{endYear}'
            if not os.path.exists(outputFileName):
                monthValues = constants.monthDictionary[season]
                dsSeason = compute_climatology(ds, monthValues, calendar, maskVaries=False)
                write_netcdf(dsSeason, outputFileName)
            ds = open_mpas_dataset(fileName=outputFileName,
                                   calendar=calendar,
                                   variableList=variableList,
                                   timeVariableNames=None)
            #TODO
            #ds = _multiply_ssh_by_area(ds)

            #TODO add region specification
            #ds.isel(nRegions=self.regionIndex))

            fields = [ds[variableList[0]]]
            #lineColors = [config.get('histogram', 'mainColor')]
            lineWidths = [3]
            legendText = [mainRunName]
            #TODO add later
            #if plotControl:
            #    fields.append(refData.isel(nRegions=self.regionIndex))
            #    lineColors.append(config.get('histogram', 'controlColor'))
            #    lineWidths.append(1.2)
            #    legendText.append(controlRunName)
            if config.has_option('histogramSSH', 'titleFontSize'):
                titleFontSize = config.getint('histogramSSH',
                                              'titleFontSize')
            else:
                titleFontSize = None

            if config.has_option('histogramSSH', 'defaultFontSize'):
                defaultFontSize = config.getint('histogramSSH',
                                                'defaultFontSize')
            else:
                defaultFontSize = None



            histogram_analysis_plot(config, fields, calendar=calendar,
                                    title=title, xlabel=xLabel, ylabel=yLabel,
                                    lineColors=lineColors,
                                    lineWidths=lineWidths, legendText=legendText, titleFontSize=titleFontSize, defaultFontSize=defaultFontSize)

            savefig(outFileName, config)

            #TODO should this be in the outer loop instead?
            caption = 'Normalized probability density function for SSH climatologies in the {} Region'.format(title)
            write_image_xml(
                config=config,
                filePrefix=filePrefix,
                componentName='Ocean',
                componentSubdirectory='ocean',
                galleryGroup='Histograms',
                groupLink='histogramSSH',
                gallery='SSH Histogram',
                thumbnailDescription=title,
                imageDescription=caption,
                imageCaption=caption)


#    def _multiply_ssh_by_area(self, ds):
#
#        """
#        Compute a time series of the global mean water-column thickness.
#        """
#
#        #TODO have not yet resolved whether we need mpasHistogramTask
#        restartFileName = \
#            mpasHistogramTask.runStreams.readpath('restart')[0]
#
#        dsRestart = xarray.open_dataset(restartFileName)
#
#        #TODO load seaIceArea for sea ice histograms
#        #landIceFraction = dsRestart.landIceFraction.isel(Time=0)
#        areaCell = dsRestart.areaCell
#
#        # for convenience, rename the variables to simpler, shorter names
#        ds = ds.rename(self.variableDict)
#
#        ds['sshAreaCell'] = \
#            ds['ssh'] / areaCell
#        ds.sshAreaCell.attrs['units'] = 'm^2'
#        ds.sshAreaCell.attrs['description'] = \
#            'Sea-surface height multiplied by the cell area'
#
#        return ds
