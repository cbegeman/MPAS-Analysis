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
Instructions for creating a new analysis task:
1. create a new task by copying this file to the appropriate folder (ocean,
   sea_ice, etc.) and modifying it as described below.  Take a look at
   mpas_analysis/shared/analysis_task.py for additional guidance.
2. note, no changes need to be made to mpas_analysis/shared/analysis_task.py
3. modify default.cfg (and possibly any machine-specific config files in
   configs/<machine>)
4. import new analysis task in mpas_analysis/<component>/__init__.py
5. add new analysis task to mpas_analysis/__main__.py under
   build_analysis_list:
      analyses.append(<component>.MyTask(config, myArg='argValue'))
   This will add a new object of the MyTask class to a list of analysis tasks
   created in build_analysis_list.  Later on in run_task, it will first
   go through the list to make sure each task needs to be generated
   (by calling check_generate, which is defined in AnalysisTask), then, will
   call setup_and_check on each task (to make sure the appropriate AM is on
   and files are present), and will finally call run on each task that is
   to be generated and is set up properly.

Don't forget to remove this docstring. (It's not needed.)
"""
# Author
# -------
# Carolyn Begeman

# import python modules here
from distutils.spawn import find_executable
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import xarray as xr

# import mpas_analysis module here (those with relative paths starting with
# dots)
from mpas_analysis.shared.analysis_task import AnalysisTask
from mpas_analysis.shared.constants import constants
from mpas_analysis.shared.io import open_mpas_dataset
from mpas_analysis.shared.io.utility import build_config_full_path, \
    make_directories, get_files_year_month, decode_strings
from mpas_analysis.shared.plot import timeseries_analysis_plot, savefig
from mpas_analysis.shared.html import write_image_xml


class ConservationTask(AnalysisTask):
    """
    <Briefly describe the analysis task here.  Just a 1-2 sentence description
    of what the task does should be sufficient.>
    """

    # Authors
    # -------
    # <List of authors>

    # This function is the "constructor", which is called when you want to
    # create a new object from your class.  Typically, it will just store a few
    # useful pieces of information like the name of the task, the component and
    # maybe some tags describing the task.
    # Keep self and config arguments.
    # self is a way to access the object that the class ConservationTask is describing,
    #    letting you access member variables and methods (member functions of a
    #    class), as you will see in the examples below.  All methods in a
    #    python class start with the argument self, which is not included in
    #    the list of arguments when you call a method of an object (because it
    #    is always included automatically).
    # config is an mpas_tools.config.MpasConfigParser object that can be used
    #    to get configuration options stored in default.cfg or a custom config
    #    file specific to a given simulation.  See examples below or in
    #    existing analysis tasks.
    # myArg should either be modified or removed. An example might be if you
    #    changed "myArg" to be "fieldName" (probably with no default value).
    #    Then, you would store it in self.fieldName so you could access it
    #    later in the script (in setup_and_check, run, or one of your helper
    #    methods).  In this example, you would have:
    #        def __init__(self, config, fieldName):...
    #    and yu would then make a new task something like this:
    #        myTask = ConservationTask(config, fieldName='seaIceArea')
    def __init__(self, config):
        """
        Construct the analysis task.
        <Add any additional description of what happens during construction>

        Parameters
        ----------
        config :  mpas_tools.config.MpasConfigParser
            Contains configuration options
        """
        # Authors
        # -------
        # Carolyn Begeman

        # first, call the constructor from the base class (AnalysisTask).
        # Modify ConservationTask, "myTask", "component" and ['tag1', 'tag2'] below:
        # taskName is the same as the class name but with a lowercase letter
        # componentName is one of 'ocean' or 'seaIce' (same as the name of
        #     the folder where the task resides)
        # tags are some useful names describing the task ('timeSeries',
        #     'climatology', 'horizontalMap', 'index', 'transect', etc.) that
        #     can be used in the 'generate' config option, e.g.
        #     'all_climatology' of 'no_transect'.  Tasks that include computing
        # time series, indices and/or climatologies should include
        # 'timeSeries', 'index' and/or 'climatology' tags, as these are helpful
        # used to compute and update the start and end dates of each of these
        # analysis tasks as part of setting up the task.
        #
        # super(ConservationTask, self).<method>() is a way of calling
        # AnalysisTask.<method>, since AnalysisTask is the "super" or parent
        # class of ConservationTask.  In this case, we first call Analysis.__init__(...)
        # before doing our own initialization.
        super(ConservationTask, self).__init__(
            config=config,
            taskName='oceanConservation',
            componentName='ocean',
            tags=['timeSeries', 'conservation'])

    # this function will be called to figure out if the analysis task should
    # run.  It should check if the input arguments to the task are supported,
    # if appropriate analysis member(s) (AMs) were turned on, if the necessary
    # output files exist, etc.  If there is a problem, an exception should be
    # raised (see the example below) so the task will not be run.
    def setup_and_check(self):
        """
        Perform steps to set up the analysis and check for errors in the setup.

        Raises
        ------
        ValueError: <if myArg has an invalid value; modify as needed>
        """
        # Authors
        # -------
        # Carolyn Begeman

        super(ConservationTask, self).setup_and_check()

        self.check_analysis_enabled(
            analysisOptionName='config_am_conservationcheck_enable',
            raiseException=True)

        config = self.config
        baseDirectory = build_config_full_path(
            config, 'output', 'conservationSubdirectory')

        make_directories(baseDirectory)

        self.outputFile = f'{baseDirectory}/{self.fullTaskName}.nc'

        # get a list of conservationCheck output files from the streams file,
        # reading only those that are between the start and end dates

        # the run directory contains the restart files
        self.runDirectory = build_config_full_path(self.config, 'input',
                                                   'runSubdirectory')
        # if the history directory exists, use it; if not, fall back on
        # runDirectory
        self.historyDirectory = build_config_full_path(
            self.config, 'input',
            '{}HistorySubdirectory'.format(self.componentName),
            defaultPath=self.runDirectory)

        self.startYear = self.config.getint('conservation', 'startYear')
        self.endYear = self.config.getint('conservation', 'endYear')
        self.inputFiles = sorted(self.historyStreams.readpath(
            'conservationCheckOutput',
            startDate=f'{self.startYear:04d}-01-01_00:00:00',
            endDate=f'{self.endYear:04d}-01-01_00:00:00',
            calendar=self.calendar))

        if len(self.inputFiles) == 0:
            raise IOError(f'No files were found matching {self.inputFile}')

        with xr.open_dataset(self.inputFiles[0]) as ds:
            self.allVariables = list(ds.data_vars.keys())

        # Each analysis task generates one or more plots and writes out an
        # associated xml file for each plot.  Once all tasks have finished,
        # the "main" task will run through all the tasks and look at
        # xmlFileNames to find out what XML files were written out.  Each task
        # should provide a list of files in the order that the corresponding
        # images should appear on the webpage.

        # Note: because of the way parallel tasks are handled in MPAS-Analysis,
        # we can't be sure that run_task() will be called (it might be
        # launched as a completely separate process) so it is not safe to store
        # a list of xml files from within run_task(). The recommended
        # procedure is to create a list of XML files here during
        # setup_and_check() and possibly use them during run_task()

        self.xmlFileNames = []

        # we also show how to store file prefixes for later use in creating
        # plots
        self.filePrefixes = {}
        self.variableList = {}


        # plotParameters is a list of parameters, a stand-ins for whatever
        # you might want to include in each plot name, for example, seasons or
        # types of observation.
        mainRunName = self.config.get('runs', 'mainRunName')
        referenceRunName = \
            config.get('runs', 'preprocessedReferenceRunName')
        referenceInputDirectory = config.get('oceanPreprocessedReference',
                                             'baseDirectory')

        self.plotTypes = self.config.getexpression('conservation', 'plotTypes')
        self.masterVariableList = {'total_mass_flux': ['netMassFlux'],
                                   'total_mass_change': ['netMassChange'],
                                   # 'land_ice_ssh_change': ['landIceSshChange'],
                                   'land_ice_mass_change': ['landIceMassChange'],
                                   'land_ice_mass_flux': ['landIceMassFlux'],
                                   'land_ice_mass_flux_components': ['accumulatedIcebergFlux',
                                                                     'accumulatedLandIceFlux',
                                                                     'accumulatedRemovedRiverRunoffFlux',
                                                                     'accumulatedRemovedIceRunoffFlux']}
        # for each derived variable, which source variables are needed
        self.derivedVariableList = {'netMassChange': ['netMassFlux'],
                                    # 'landIceSshChange': [],
                                    'landIceMassFlux': ['accumulatedIcebergFlux',
                                                          'accumulatedLandIceFlux',
                                                          'accumulatedRemovedRiverRunoffFlux',
                                                          'accumulatedRemovedIceRunoffFlux'],
                                    'landIceMassChange': ['accumulatedIcebergFlux',
                                                          'accumulatedLandIceFlux',
                                                          'accumulatedRemovedRiverRunoffFlux',
                                                          'accumulatedRemovedIceRunoffFlux']}
        for plot_type in self.plotTypes:
            if plot_type not in self.masterVariableList.keys():
                raise ValueError(f'plot type {plot_type} not supported')
            filePrefix = f'conservation_{mainRunName}_{plot_type}_' \
                         f'years{self.startYear:04d}-{self.endYear:04d}'
            self.xmlFileNames.append('{}/{}.xml'.format(self.plotsDirectory,
                                                        filePrefix))
            self.filePrefixes[plot_type] = filePrefix
            self.variableList[plot_type] = self._add_variables(self.masterVariableList[plot_type])

    def run_task(self):
        """
        The main method of the task that performs the analysis task.
        """
        # Authors
        # -------
        # <List of authors>

        all_plots_variable_list = []
        for plot_type in self.plotTypes:
            for varname in self.variableList[plot_type]:
                all_plots_variable_list.append(varname)
        self.logger.info(all_plots_variable_list)
        self._compute_time_series_with_ncrcat(all_plots_variable_list)
        for plot_type in self.plotTypes:
            self._make_plot(plot_type)

    def _add_variables(self, target_variable_list):
        """
        Add one or more variables to extract as a time series.

        Parameters
        ----------
        variableList : list of str
            A list of variable names in ``conservationCheck`` to be
            included in the time series

        Raises
        ------
        ValueError
            if this funciton is called before this task has been set up (so
            the list of available variables has not yet been set) or if one
            or more of the requested variables is not available in the
            ``conservationCheck`` output.
        """
        # Authors
        # -------
        # Xylar Asay-Davis

        # These are all of the plotTypes that are supported
        # TODO add check to setup_and_check
        variable_list = []
        if self.allVariables is None:
            raise ValueError('add_variables() can only be called after '
                             'setup_and_check() in ConservationTask.\n'
                             'Presumably tasks were added in the wrong order '
                             'or add_variables() is being called in the wrong '
                             'place.')

        for variable in target_variable_list:
            if variable not in self.allVariables and \
                        variable not in self.derivedVariableList.keys():
                raise ValueError(
                    '{} is not available in conservationCheck'
                    'output:\n{}'.format(variable, self.allVariables))

            if variable in self.allVariables and variable not in variable_list:
                variable_list.append(variable)
            if variable in self.derivedVariableList.keys() and \
                   variable not in variable_list:
                for var in self.derivedVariableList[variable]:
                    variable_list.append(var)

        return variable_list

    def _make_plot(self, plot_type):
        """
        Comarison with a reference run is not yet supported

        Parameters
        ----------
        plot_type: str
            The type of plot to generate from conservationCheck variables
        """
        config = self.config
        filePrefix = self.filePrefixes[plot_type]
        outFileName = f'{self.plotsDirectory}/{filePrefix}.png'
        titles = {}
        titles['total_mass_flux'] = 'Total mass flux'
        titles['total_mass_change'] = 'Total mass anomaly'
        titles['land_ice_mass_change'] = 'Mass anomaly due to land ice fluxes'
        titles['land_ice_mass_flux_components'] = 'Mass fluxes from land ice'
        y_labels = {}
        y_labels['total_mass_flux'] = 'Mass flux (Gt/yr)'
        y_labels['total_mass_change'] = 'Mass (Gt)'
        y_labels['land_ice_mass_change'] = 'Mass (Gt)'
        y_labels['land_ice_mass_flux_components'] = 'Mass flux (Gt/yr)'

        #ds_variables = []
        #for varname in self.variableList[plot_type]:
        #    ds_variables.append(varname)

        self.logger.info(f'  Open conservation file {self.outputFile}...')
        ds = open_mpas_dataset(fileName=self.outputFile,
                               calendar=self.calendar,
                               variableList=self.variableList[plot_type],
                               timeVariableNames='xtime',
                               startDate=f'{self.startYear:04d}-01-01_00:00:00',
                               endDate=f'{self.endYear:04d}-01-01_00:00:00')

        if referenceRunName != 'None':
            inFilesPreprocessed = f'{referenceInputDirectory}/{self.fullTaskName}.nc'
            self.logger.info('  Load in conservation for a preprocessed reference '
                             'run {inFilesPreprocessed}...')
            ds_ref = open_mpas_dataset(fileName=inFilesPreprocessed,
                                   calendar=self.calendar,
                                   variableList=self.variableList[plot_type],
                                   timeVariableNames='xtime')
            yearEndPreprocessed = days_to_datetime(ds_ref.Time.max(),
                                                   calendar=calendar).year
            if self.startYear <= yearEndPreprocessed:
                timeStart = date_to_days(year=self.startYear, month=1, day=1,
                                         calendar=calendar)
                timeEnd = date_to_days(year=self.endYear, month=12, day=31,
                                       calendar=calendar)
                dsPreprocessedTimeSlice = \
                    dsPreprocessed.sel(Time=slice(timeStart, timeEnd))
            else:
                self.logger.warning('Preprocessed time series ends before the '
                                    'timeSeries startYear and will not be '
                                    'plotted.')
                referenceRunName = 'None'

        # make the plot
        self.logger.info('  Make conservation plots...')
        xLabel = 'Time (years)'
        title = titles[plot_type]
        yLabel = y_labels[plot_type]
        lineStylesBase = ['-', '--', '-.', ':']
        # get xtime for all variables
        # make sure xtime is in decimal years

        fields = []
        legendText = []
        lineColors = []
        lineStyles = []
        for index, varname in enumerate(self.masterVariableList[plot_type]):
            if varname in self.derivedVariableList:
                variable = self._derive_variable(ds, varname)
            else:
                variable = ds[varname]
            if 'Removed' in varname:
                variable = -variable
            if 'mass_flux' in plot_type:
                variable = variable * 1e-12 * constants.sec_per_year  # convert to Gt/yr
            fields.append(variable)
            legendText.append(varname)
            lineColors.append(config.get('timeSeries', 'mainColor'))
            lineStyles.append(lineStylesBase[index])
            if referenceRunName != 'None':
                if varname in self.derivedVariableList:
                    variable = self._derive_variable(ds_ref, varname)
                else:
                    variable = ds_ref[varname]
                if 'Removed' in varname:
                    variable = -variable
                if 'mass_flux' in plot_type:
                    variable = variable * 1e-12 * constants.sec_per_year  # convert to Gt/yr
                fields.append(variable)
                legendText.append(varname)
                lineColors.append(config.get('timeSeries', 'controlColor'))
                lineStyles.append(lineStylesBase[index])

        lineWidths = [3 for i in fields]
        if config.has_option('conservation', 'movingAveragePoints'):
            movingAveragePoints = config.getint('conservation',
                                                'movingAveragePoints')
        else:
            movingAveragePoints = None

        if config.has_option('conservation', 'firstYearXTicks'):
            firstYearXTicks = config.getint('conservation',
                                            'firstYearXTicks')
        else:
            firstYearXTicks = None

        if config.has_option('conservation', 'yearStrideXTicks'):
            EearStrideXTicks = config.getint('conservation',
                                             'yearStrideXTicks')
        else:
            yearStrideXTicks = None

        timeseries_analysis_plot(config, fields, calendar=self.calendar,
                                 title=title, xlabel=xLabel, ylabel=yLabel,
                                 movingAveragePoints=movingAveragePoints,
                                 lineColors=lineColors,
                                 lineStyles=lineStyles[:len(fields)],
                                 lineWidths=lineWidths,
                                 legendText=legendText,
                                 firstYearXTicks=firstYearXTicks,
                                 yearStrideXTicks=yearStrideXTicks)

        # save the plot to the output file
        plt.savefig(outFileName)

        # here's an example of how you would create an XML file for this plot
        # with the appropriate entries.  Some notes:
        # * Gallery groups typically represent all the analysis from a task,
        #   or sometimes from multiple tasks
        # * A gallery might be for just for one set of observations, one
        #   season, etc., depending on what makes sense
        # * Within each gallery, there is one plot for each value in
        #   'plotParameters', with a corresponding caption and short thumbnail
        #   description
        caption = '' # TODO captions[plot_type]
        write_image_xml(
            config=self.config,
            filePrefix=filePrefix,
            componentName='Ocean',
            componentSubdirectory='ocean',
            galleryGroup='Time Series',
            groupLink='timeseries',
            gallery=title,
            thumbnailDescription=plot_type,
            imageDescription=caption,
            imageCaption=caption)

    def _derive_variable(self, ds, varname):

        if varname == 'netMassChange':
            # Convert from kg/s to Gt/s
            mass_flux = ds['netMassFlux'] * 1e-12
            # Assume that the frequency of output is monthly
            dt = constants.sec_per_month
            # Convert from Gt/yr to Gt
            derived_variable = mass_flux.cumsum(axis=0) * dt
        elif varname == 'landIceMassChange':
            land_ice_mass_flux = self._derive_variable(ds, 'landIceMassFlux')
            # Convert from kg/s to Gt/s
            land_ice_mass_flux = land_ice_mass_flux * 1e-12
            # Assume that the frequency of output is monthly
            dt = constants.sec_per_month
            # Convert from Gt/yr to Gt
            derived_variable = land_ice_mass_flux.cumsum(axis=0) * dt

        elif varname == 'landIceMassFlux':
            # Here, keep units as kg/s because the conversion to Gt/yr will happen later
            derived_variable = ds['accumulatedIcebergFlux'] + \
                ds['accumulatedLandIceFlux'] + \
                -ds['accumulatedRemovedRiverRunoffFlux'] + \
                -ds['accumulatedRemovedIceRunoffFlux']
        else:
            raise ValueError(f'Attempted to derive non-supported variable {varname}')
        # Conversion to ssh signal is not yet supported because we need to get
        # area from a different file
        # elif varname == 'landIceSshChange':
        #     land_ice_mass_change = self._derive_variable(ds, 'landIceMassChange')
        #     # Convert from to Gt to kg
        #     land_ice_mass_change = land_ice_mass_change * 1e12
        #     ds_ts = xr.open_dataset(f'{run_path}/{file_prefix}.{ts_prefix}.{date_suffix}')
        #     A = ds_ts.timeMonthly_avg_areaCellGlobal
        #     # Convert from to kg to mm
        #     rho = self.namelist.getfloat('config_density0')
        #     derived_variable = land_ice_mass_change * 1.0e3 / (rho * A)

        return derived_variable

    def _compute_time_series_with_ncrcat(self, variable_list):

        """
        Uses ncrcat to extact time series from conservationCheckOutput files

        Raises
        ------
        OSError
            If ``ncrcat`` is not in the system path.

        Author
        ------
        Xylar Asay-Davis
        """

        if find_executable('ncrcat') is None:
            raise OSError('ncrcat not found. Make sure the latest nco '
                          'package is installed: \n'
                          'conda install nco\n'
                          'Note: this presumes use of the conda-forge '
                          'channel.')

        inputFiles = self.inputFiles
        append = False
        if os.path.exists(self.outputFile):
            # make sure all the necessary variables are also present
            with xr.open_dataset(self.outputFile) as ds:
                if ds.sizes['Time'] == 0:
                    updateSubset = False
                else:
                    updateSubset = True
                    for variableName in variable_list:
                        if variableName not in ds.variables:
                            updateSubset = False
                            break

                if updateSubset:
                    # add only input files wiht times that aren't already in
                    # the output file

                    append = True

                    fileNames = sorted(self.inputFiles)
                    inYears, inMonths = get_files_year_month(
                        fileNames, self.historyStreams,
                        'conservationCheckOutput')

                    inYears = np.array(inYears)
                    inMonths = np.array(inMonths)
                    totalMonths = 12 * inYears + inMonths

                    dates = decode_strings(ds.xtime)

                    lastDate = dates[-1]

                    lastYear = int(lastDate[0:4])
                    lastMonth = int(lastDate[5:7])
                    lastTotalMonths = 12 * lastYear + lastMonth

                    inputFiles = []
                    for index, inputFile in enumerate(fileNames):
                        if totalMonths[index] > lastTotalMonths:
                            inputFiles.append(inputFile)

                    if len(inputFiles) == 0:
                        # nothing to do
                        return
                else:
                    # there is an output file but it has the wrong variables
                    # so we need ot delete it.
                    self.logger.warning('Warning: deleting file {} because '
                                        'it is empty or some variables were '
                                        'missing'.format(self.outputFile))
                    os.remove(self.outputFile)

        variableList = variable_list + ['xtime']

        args = ['ncrcat', '-4', '--no_tmp_fl',
                '-v', ','.join(variableList)]

        if append:
            args.append('--record_append')

        printCommand = '{} {} ... {} {}'.format(' '.join(args), inputFiles[0],
                                                inputFiles[-1],
                                                self.outputFile)
        args.extend(inputFiles)
        args.append(self.outputFile)

        self.logger.info('running: {}'.format(printCommand))
        for handler in self.logger.handlers:
            handler.flush()

        process = subprocess.Popen(args, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if stdout:
            stdout = stdout.decode('utf-8')
            for line in stdout.split('\n'):
                self.logger.info(line)
        if stderr:
            stderr = stderr.decode('utf-8')
            for line in stderr.split('\n'):
                self.logger.error(line)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode,
                                                ' '.join(args))
