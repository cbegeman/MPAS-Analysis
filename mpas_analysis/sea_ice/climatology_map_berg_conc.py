
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import xarray as xr

from mpas_analysis.shared import AnalysisTask

from mpas_analysis.shared.climatology import RemapMpasClimatologySubtask, \
    RemapObservedClimatologySubtask

from mpas_analysis.sea_ice.plot_climatology_map_subtask import \
    PlotClimatologyMapSubtask

from mpas_analysis.shared.io.utility import build_config_full_path

from mpas_analysis.shared.grid import LatLonGridDescriptor


class ClimatologyMapIcebergConc(AnalysisTask):  # {{{
    """
    An analysis task for comparison of iceberg concentration against
    observations
    """
    # Authors
    # -------
    # Darin Comeau, Xylar Asay-Davis

    def __init__(self, config, mpasClimatologyTask, hemisphere,
                 refConfig=None):  # {{{
        """
        Construct the analysis task.

        Parameters
        ----------
        config :  ``MpasAnalysisConfigParser``
            Configuration options

        mpasClimatologyTask : ``MpasClimatologyTask``
            The task that produced the climatology to be remapped and plotted

        hemisphere : {'NH', 'SH'}
            The hemisphere to plot

        refConfig :  ``MpasAnalysisConfigParser``, optional
            Configuration options for a reference run (if any)
        """
        # Authors
        # -------
        # Darin Comeau, Xylar Asay-Davis

        taskName = 'climatologyMapIcebergConc{}'.format(hemisphere)

        fieldName = 'IcebergConc'
        # call the constructor from the base class (AnalysisTask)
        super(ClimatologyMapIcebergConc, self).__init__(
                config=config, taskName=taskName,
                componentName='seaIce',
                tags=['icebergs', 'climatology', 'horizontalMap', fieldName])

        mpasFieldName = 'timeMonthly_avg_bergAreaCell'
        iselValues = None

        sectionName = taskName

        if hemisphere == 'NH':
            hemisphereLong = 'Northern'
        else:
            hemisphereLong = 'Southern'

        # read in what seasons we want to plot
        seasons = config.getExpression(sectionName, 'seasons')

        if len(seasons) == 0:
            raise ValueError('config section {} does not contain valid list '
                             'of seasons'.format(sectionName))

        comparisonGridNames = config.getExpression(sectionName,
                                                   'comparisonGrids')

        if len(comparisonGridNames) == 0:
            raise ValueError('config section {} does not contain valid list '
                             'of comparison grids'.format(sectionName))

        # the variable self.mpasFieldName will be added to mpasClimatologyTask
        # along with the seasons.
        remapClimatologySubtask = RemapMpasClimatologySubtask(
            mpasClimatologyTask=mpasClimatologyTask,
            parentTask=self,
            climatologyName='{}{}'.format(fieldName, hemisphere),
            variableList=[mpasFieldName],
            comparisonGridNames=comparisonGridNames,
            seasons=seasons,
            iselValues=iselValues)

        if refConfig is None:
            refTitleLabel = 'Observations (Altiberg)'
            galleryName = 'Observations: Altiberg'
            diffTitleLabel = 'Model - Observations'
            refFieldName = 'icebergConc'
            obsFileName = build_config_full_path(
                    config, 'icebergObservations',
                    'concentration{}'.format(hemisphere))

            remapObservationsSubtask = RemapAltibergConcClimatology(
                    parentTask=self, seasons=seasons,
                    fileName=obsFileName,
                    outFilePrefix='{}{}'.format(refFieldName,
                                                hemisphere),
                    comparisonGridNames=comparisonGridNames)
            self.add_subtask(remapObservationsSubtask)

        else:
            refRunName = refConfig.get('runs', 'mainRunName')
            galleryName = None
            refTitleLabel = 'Ref: {}'.format(refRunName)
            refFieldName = mpasFieldName
            diffTitleLabel = 'Main - Reference'

            remapObservationsSubtask = None

        for season in seasons:
            for comparisonGridName in comparisonGridNames:

                imageDescription = \
                    '{} Climatology Map of {}-Hemisphere Iceberg ' \
                    'Concentration.'.format(season, hemisphereLong)
                imageCaption = imageDescription
                galleryGroup = \
                    '{}-Hemisphere Iceberg Concentration'.format(
                            hemisphereLong)
                # make a new subtask for this season and comparison grid
                subtask = PlotClimatologyMapSubtask(
                        self, hemisphere, season, comparisonGridName,
                        remapClimatologySubtask, remapObservationsSubtask,
                        refConfig)

                subtask.set_plot_info(
                        outFileLabel='bergconc{}'.format(hemisphere),
                        fieldNameInTitle='Iceberg concentration',
                        mpasFieldName=mpasFieldName,
                        refFieldName=refFieldName,
                        refTitleLabel=refTitleLabel,
                        diffTitleLabel=diffTitleLabel,
                        unitsLabel=r'%',
                        imageDescription=imageDescription,
                        imageCaption=imageCaption,
                        galleryGroup=galleryGroup,
                        groupSubtitle=None,
                        groupLink='{}_conc'.format(hemisphere.lower()),
                        galleryName=galleryName,
                        maskValue=None)

                self.add_subtask(subtask)

        # }}}

    # }}}


class RemapAltibergConcClimatology(RemapObservedClimatologySubtask):  # {{{
    """
    A subtask for reading and remapping iceberg concentration from Altiberg
    observations
    """
    # Authors
    # -------
    # Darin Comeau, Xylar Asay-Davis

    def get_observation_descriptor(self, fileName):  # {{{
        '''
        get a MeshDescriptor for the observation grid

        Parameters
        ----------
        fileName : str
            observation file name describing the source grid

        Returns
        -------
        obsDescriptor : ``MeshDescriptor``
            The descriptor for the observation grid
        '''
        # Authors
        # -------
        # Darin Comeau, Xylar Asay-Davis

        # create a descriptor of the observation grid using the lat/lon
        # coordinates
        obsDescriptor = LatLonGridDescriptor.read(fileName=fileName,
                                                  latVarName='latitude',
                                                  lonVarName='longitude')
        return obsDescriptor  # }}}

    def build_observational_dataset(self, fileName):  # {{{
        '''
        read in the data sets for observations, and possibly rename some
        variables and dimensions

        Parameters
        ----------
        fileName : str
            observation file name

        Returns
        -------
        dsObs : ``xarray.Dataset``
            The observational dataset
        '''
        # Authors
        # -------
        # Darin Comeau, Xylar Asay-Davis

        dsObs = xr.open_dataset(fileName)
        dsObs.rename({'probability': 'icebergConc', 'time': 'Time'},
                     inplace=True)
        dsObs.coords['month'] = dsObs['Time.month']
        dsObs.coords['year'] = dsObs['Time.year']
        dsObs = dsObs.transpose('Time', 'latitude', 'longitude')

        return dsObs  # }}}
    # }}}


# vim: foldmethod=marker ai ts=4 sts=4 et sw=4 ft=python
