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

        baseDirectory = build_config_full_path(
            config, 'output', 'wmtSubdirectory')
        if not os.path.exists(baseDirectory):
            make_directories(baseDirectory)

        for regionGroup in self.regionGroups:
            groupObsDicts = {}
            mpasMasksSubtask = regionMasksTask.add_mask_subtask(
                regionGroup=regionGroup)
            regionNames = mpasMasksSubtask.expand_region_names(
                self.regionNames)

            regionGroupSuffix = regionGroup.replace(' ', '_')
            filePrefix = f'wmt_{regionGroupSuffix}'

            for regionName in regionNames:
                # Generate histogram plots
                plotRegionSubtask = PlotRegionWmtSubtask(
                    self, regionGroup, regionName, controlConfig,
                    sectionName, filePrefix, mpasMasksSubtask)
                plotRegionSubtask.run_after(mpasMasksSubtask)
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

        filenames = glob.glob(
                f'{base_directory}/{self.filePrefix}_{self.regionName}_*' \
                'wmt.nc')

        ds_bins = filenames[0]
        ds_values = []
        for filename in filenames:
            with xr.open_dataset(filename) as ds:
                ds_values.append(ds)

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
