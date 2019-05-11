#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.

Public classes:

  - :class:`DDF`:
      Distributed DataFrame (DDF), the abstraction of this library.
"""

from pycompss.api.api import compss_wait_on

from ddf_library.ddf_base import DDFSketch
from ddf_library.context import COMPSsContext
from ddf_library.utils import generate_info, concatenate_pandas

import pandas as pd
import numpy as np
import copy


__all__ = ['DDF', 'generate_info']


class DDF(DDFSketch):
    """
    Distributed DataFrame Handler.

    Should distribute the data and run tasks for each partition.
    """

    def __init__(self, **kwargs):
        super(DDF, self).__init__()

        task_list = kwargs.get('task_list', None)
        last_uuid = kwargs.get('last_uuid', 'init')
        self.settings = kwargs.get('settings', {'input': 0})

        self.partitions = list()

        if last_uuid != 'init':

            self.task_list = copy.deepcopy(task_list)
            self.task_list.append(last_uuid)

        else:
            last_uuid = self._ddf_add_task('init', 'COMPLETED', self.OPT_OTHER,
                                           None, [], None, 0, info=None)

            self.task_list = list()
            self.task_list.append(last_uuid)

        self.last_uuid = last_uuid

    def __str__(self):
        return "DDF object."

    def load_text(self, filename, num_of_parts=4, header=True,
                  sep=',', dtype=None, na_values=None, storage='hdfs',
                  host='localhost', port=9000, distributed=False):
        """
        Create a DDF from a common file system or from HDFS.

        :param filename: Input file name;
        :param num_of_parts: number of partitions (default, 4);
        :param header: Use the first line as DataFrame header (default, True);
        :param dtype: Type name or dict of column (default, 'str'). Data type
         for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32, ‘c’:
         ‘Int64’} Use str or object together with suitable na_values settings
         to preserve and not interpret dtype;
        :param sep: separator delimiter (default, ',');
        :param na_values: A list with the all nan characters. Default list:
         ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
         '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan']
        :param storage: *'hdfs'* to use HDFS as storage or *'fs'* to use the
         common file system;
        :param distributed: if the absolute path represents a unique file or
         a folder with multiple files;
        :param host: Namenode host if storage is `hdfs` (default, 'localhost');
        :param port: Port to Namenode host if storage is `hdfs` (default, 9000);
        :return: DDF.

        ..see also: Visit this `link <https://docs.scipy.org/doc/numpy-1.15
        .0/reference/arrays.dtypes.html>`__ to more information about dtype.

        :Example:

        >>> ddf1 = DDF().load_text('/titanic.csv', num_of_parts=4)
        """
        if storage not in ['hdfs', 'fs']:
            raise Exception('`hdfs` and `fs` storage are supported.')

        from .functions.etl.read_data import DataReader

        data_reader = DataReader(filename, nfrag=num_of_parts,
                                 format='csv', storage=storage,
                                 distributed=distributed,
                                 dtype=dtype, separator=sep, header=header,
                                 na_values=na_values, host=host, port=port)

        if storage is 'fs':

            if distributed:
                blocks = data_reader.get_blocks()
                COMPSsContext.tasks_map[self.last_uuid]['function'] = \
                    {0: blocks}

                def reader(block, params):
                    return data_reader.transform_fs_distributed(block, params)

                new_state_uuid = self._generate_uuid()
                COMPSsContext.tasks_map[new_state_uuid] = \
                    {'name': 'load_text-file_in',
                     'status': 'WAIT',
                     'optimization': self.OPT_SERIAL,
                     'function': [reader, {}],
                     'output': 1,
                     'input': 0,
                     'parent': [self.last_uuid]
                     }

            else:

                result, info = data_reader.transform_fs_single()

                new_state_uuid = self._generate_uuid()
                COMPSsContext.tasks_map[new_state_uuid] = \
                    {'name': 'load_text',
                     'status': 'COMPLETED',
                     'optimization': self.OPT_OTHER,
                     'function': {0: result},
                     'output': 1,
                     'input': 0,
                     'parent': [self.last_uuid]
                     }

                COMPSsContext.schemas_map[new_state_uuid] = {0: info}
        else:
            blocks = data_reader.get_blocks()

            COMPSsContext.tasks_map[self.last_uuid]['function'] = \
                {0: blocks}

            def reader(block, params):
                return data_reader.transform_hdfs(block, params)

            new_state_uuid = self._generate_uuid()
            COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'load_text',
                 'status': 'WAIT',
                 'optimization': self.OPT_SERIAL,
                 'function': [reader, {}],
                 'output': 1,
                 'input': 0,
                 'parent': [self.last_uuid]
                 }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def parallelize(self, df, num_of_parts=4):
        """
        Distributes a DataFrame into DDF.

        :param df: DataFrame input
        :param num_of_parts: number of partitions
        :return: DDF

        :Example:

        >>> ddf1 = DDF().parallelize(df)
        """

        from .functions.etl.parallelize import parallelize

        result, info = parallelize(df, num_of_parts)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.schemas_map[new_state_uuid] = {0: info}
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'parallelize',
             'status': 'COMPLETED',
             'optimization': self.OPT_OTHER,
             'function': {0: result},
             'output': 1, 'input': 0,
             'parent': [self.last_uuid]
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def load_shapefile(self, shp_path, dbf_path, polygon='points',
                       attributes=None, num_of_parts=4):
        """
        Reads a shapefile using the shp and dbf file.

        :param shp_path: Path to the shapefile (.shp)
        :param dbf_path: Path to the shapefile (.dbf)
        :param polygon: Alias to the new column to store the
                polygon coordenates (default, 'points');
        :param attributes: List of attributes to keep in the dataframe,
                empty to use all fields;
        :param num_of_parts: The number of fragments;
        :return: DDF

        .. note:: $ pip install pyshp

        :Example:

        >>> ddf1 = DDF().load_shapefile(shp_path='/41CURITI.shp',
        >>>                             dbf_path='/41CURITI.dbf')
        """

        if attributes is None:
            attributes = []

        settings = dict()
        settings['shp_path'] = shp_path
        settings['dbf_path'] = dbf_path
        settings['polygon'] = polygon
        settings['attributes'] = attributes

        from .functions.geo import read_shapefile

        result, info = read_shapefile(settings, num_of_parts)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.schemas_map[new_state_uuid] = {0: info}
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'load_shapefile',
             'status': 'COMPLETED',
             'optimization': self.OPT_OTHER,
             'function': {0: result},
             'output': 1, 'input': 0,
             'parent': [self.last_uuid]
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def import_data(self, df_list, info=None):
        """
        Import a previous Pandas DataFrame list in DDF abstraction.
        Replace old data if DDF is not empty.

        :param df_list: DataFrame input
        :param info: (Optional) A list of columns names, data types and size
         in each partition;
        :return: DDF

        :Example:

        >>> ddf1 = DDF().import_partitions(df_list)
        """

        from .functions.etl.parallelize import import_to_ddf

        result, info = import_to_ddf(df_list, info)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.schemas_map[new_state_uuid] = {0: info}

        tmp = DDF()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'import_data',
             'status': 'COMPLETED',
             'optimization': self.OPT_OTHER,
             'function': {0: result},
             'output': 1, 'input': 0,
             'parent': [tmp.last_uuid]
             }

        tmp._set_n_input(new_state_uuid, 0)
        return DDF(task_list=tmp.task_list, last_uuid=new_state_uuid)

    def _collect(self):
        """
        #TODO Check it
        :return:
        """

        if COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
            self.partitions = \
                COMPSsContext.tasks_map[self.last_uuid]['function'][0]

        self.partitions = compss_wait_on(self.partitions)

    def cache(self):
        """
        Compute all tasks until the current state

        :return: DDF

        :Example:

        >>> ddf1.cache()
        """

        # TODO: no momento, é necessario para lidar com split
        # if COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
        #     print 'cache skipped'
        #     return self

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sync',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'parent': [self.last_uuid],
             'function': [[]]
             }

        self._set_n_input(new_state_uuid, self.settings['input'])

        tmp = DDF(task_list=self.task_list,
                  last_uuid=new_state_uuid,
                  settings=self.settings)._run_compss_context(new_state_uuid)

        return tmp

    def num_of_partitions(self):
        """
        Returns the number of data partitions (Task parallelism).

        :return: integer

        :Example:

        >>> print(ddf1.num_of_partitions())
        """

        info = self._get_info()
        size = len(info['size'])
        return size

    def balancer(self, forced=False):
        """

        """

        from .functions.etl.balancer import WorkloadBalancer
        settings = {'forced': forced}

        def task_balancer(df, settings):
            return WorkloadBalancer(settings).transform(df)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'balancer',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_balancer, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def count_rows(self, total=True):
        """
        Return a number of rows in this DDF.

        :param total: Show the total number (default) or the number over
         the fragments.
        :return: integer

        :Example:

        >>> print(ddf1.count_rows())
        """

        info = self._get_info()
        size = info['size']

        if total:
            size = sum(size)

        return size

    def cast(self, column, cast):
        """
        Change the data's type of some columns.

        Is it a Lazy function: Yes

        :param column: String or list of strings with columns to cast;
        :param cast: String or list of string with the supported types:
         'integer', 'string', 'double', 'date', 'date/time';
        :return: DDF
        """

        from .functions.etl.attributes_changer import with_column_cast

        if not isinstance(column, list):
            column = [column]

        if not isinstance(cast, list):
            cast = [cast for _ in range(len(column))]

        diff = len(cast) - len(column)
        if diff > 0:
            cast = cast[:len(column)]
        elif diff < 0:
            cast = cast + ['keep' for _ in range(diff+1)]

        settings = dict()
        settings['attributes'] = column
        settings['cast'] = cast

        def task_cast(df, params):
            return with_column_cast(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'cast',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_cast, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])

        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def add_column(self, data2, suffixes=None):
        """
        Merge two DDF, column-wise.

        Is it a Lazy function: No

        :param data2: The second DDF;
        :param suffixes: Suffixes in case of duplicated columns name
            (default, ["_l","_r"]);
        :return: DDF

        :Example:

        >>> ddf1.add_column(ddf2)
        """

        if suffixes is None:
            suffixes = ['_l', '_r']

        settings = {'suffixes': suffixes}

        from .functions.etl.add_columns import AddColumnsOperation

        def task_add_column(df, params):
            return AddColumnsOperation().transform(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'add_column',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_add_column, settings],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def group_by(self, group_by):
        """
        Computes aggregates and returns the result as a DDF.

        Is it a Lazy function: No

        :param group_by: A list of columns to be grouped;
        :return: A GroupedDFF with a set of methods for aggregations on a DDF

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).mean(['col_2']).first(['col_2'])
        """
        settings = {'groupby': group_by, 'operation': {}}
        from .groupby import GroupedDDF
        from .functions.etl.aggregation import AggregationOperation

        def task_aggregation(df, params):
            return AggregationOperation().transform(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'groupby',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_aggregation, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        tmp = DDF(task_list=self.task_list, last_uuid=new_state_uuid)

        return GroupedDDF(tmp)

    def fillna(self, subset=None, mode='VALUE', value=None):
        """
        Replace missing rows or columns by mean, median, value or mode.

        Is it a Lazy function: Yes, only if mode *"VALUE"*

        :param subset: A list of attributes to evaluate;
        :param mode: action in case of missing values: *"VALUE"* to replace by
         parameter "value" (default); *"MEDIAN"* to replace by median value;
         *"MODE"* to replace by  mode value; *"MEAN"* to replace by mean value;
        :param value: Value to be replaced (only if mode is *"VALUE"*)
        :return: DDF

        :Example:

        >>> ddf1.fillna(['col_1'], value=42)
        """

        from .functions.etl.clean_missing import FillNa

        lazy = self.OPT_OTHER
        if mode is 'VALUE':
            lazy = self.OPT_SERIAL

            if not value:
                raise Exception("It is necessary a value "
                                "when using `VALUE` mode.")

            def task_clean_missing(df, _):
                return FillNa(subset, mode, value).fill_by_value(df)

        else:

            def task_clean_missing(df, _):
                return FillNa(subset, mode=mode, value=value)\
                        .preprocessing(df)\
                        .fill_by_statistic(df)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'fill_na',
             'status': 'WAIT',
             'optimization': lazy,
             'function': [task_clean_missing, {}],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }
        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def columns(self):

        columns = self._get_info()['cols']
        return columns

    def correlation(self, col1, col2):
        """
        Calculate the Pearson Correlation Coefficient.

        When one of the standard deviations is zero so the correlation is
        undefined (NaN).

        :param col1: The name of the first column
        :param col2: The name of the second column
        :return: The result of sample covariance
        """

        if not isinstance(col1, str):
            raise ValueError("col1 should be a string.")
        if not isinstance(col2, str):
            raise ValueError("col2 should be a string.")

        df, nfrag, tmp = self._ddf_initial_setup(self)

        from ddf_library.functions.statistics.correlation import correlation

        params = {'col1': col1, 'col2': col2}
        result = correlation(df, params)

        return result

    def covariance(self, col1, col2):
        """
        Calculate the sample covariance for the given columns, specified by
        their names, as a double value.

        :param col1: The name of the first column
        :param col2: The name of the second column
        :return: The result of sample covariance
        """

        if not isinstance(col1, str):
            raise ValueError("col1 should be a string.")
        if not isinstance(col2, str):
            raise ValueError("col2 should be a string.")

        df, nfrag, tmp = self._ddf_initial_setup(self)

        from ddf_library.functions.statistics.covariance import covariance

        params = {'col1': col1, 'col2': col2}
        result = covariance(df, params)

        return result

    def cross_join(self, data2):
        """
        Returns the cartesian product with another DDF.

        :param data2: Right side of the cartesian product;
        :return: DDF.

        :Example:

        >>> ddf1.cross_join(ddf2)
        """
        from .functions.etl.cross_join import crossjoin

        def task_cross_join(df, _):
            return crossjoin(df[0], df[1])

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'cross_join',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_cross_join, {}],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def cross_tab(self, col1, col2):
        """
        Computes a pair-wise frequency table of the given columns. Also known as
        a contingency table.  The number of distinct values for each column
        should be less than 1e4. At most 1e6 non-zero pair frequencies will be
        returned.

        Is it a Lazy function: No

        :param col1: The name of the first column
        :param col2: The name of the second column
        :return: DDF

        :Example:

        >>> ddf1.cross_tab(col1='col_1', col2='col_2')
        """
        from ddf_library.functions.statistics.cross_tab import cross_tab
        settings = {'col1': col1, 'col2': col2}

        def task_cross_tab(df, params):
            return cross_tab(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'cross_tab',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_cross_tab, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def describe(self, columns=None):
        """
        Computes basic statistics for numeric and string columns. This include
        count, mean, number of NaN, stddev, min, and max.

        Is it a Lazy function: No

        :param columns: A list of columns, if no columns are given,
         this function computes statistics for all numerical or string columns;
        :return: A pandas DataFrame

        :Example:

        >>> ddf1.describe(['col_1'])
        """

        df, nfrag, tmp = self._ddf_initial_setup(self)

        from ddf_library.functions.statistics.describe import describe

        if not columns:
            columns = []

        result = describe(df, columns)

        return result

    def freq_items(self, col, support=0.01):
        """
        Finding frequent items for columns, possibly with false positives.
        Using the frequent element count algorithm described in
        “http://dx.doi.org/10.1145/762471.762473, proposed by Karp, Schenker,
        and Papadimitriou”

        Is it a Lazy function: No

        :param col: Names of the columns to calculate frequent items
        :param support: The frequency with which to consider an item 'frequent'.
         Default is 1%. The support must be greater than 1e-4.
        :return: DDF

        :Example:

        >>> ddf1.freq_items(col='col_1', support=0.01)
        """
        from ddf_library.functions.statistics.freq_items import freq_items
        settings = {'col': col, 'support': support}

        df, nfrag, tmp = self._ddf_initial_setup(self)

        result = freq_items(df, settings)

        return result

    def subtract(self, data2):
        """
        Returns a new DDF with containing rows in the first frame but not
        in the second one. This is equivalent to EXCEPT in SQL.

        Is it a Lazy function: No

        :param data2: second DDF;
        :return: DDF

        :Example:

        >>> ddf1.subtract(ddf2)
        """
        from .functions.etl.subtract import subtract

        settings = {}

        def task_difference(df, params):
            return subtract(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'subtract',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_difference, settings],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def except_all(self, data2):
        """
        Returns a new DDF with containing rows in the first frame but not
        in the second one while preserving duplicates. This is equivalent to
        EXCEPT ALL in SQL.

        Is it a Lazy function: No

        :param data2: second DDF;
        :return: DDF

        :Example:

        >>> ddf1.except_all(ddf2)
        """
        from .functions.etl.except_all import except_all

        settings = {}

        def task_except_all(df, params):
            return except_all(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'except_all',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_except_all, settings],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def distinct(self, cols):
        """
        Returns a new DDF containing the distinct rows in this DDF.

        Is it a Lazy function: No

        :param cols: subset of columns;
        :return: DDF

        :Example:

        >>> ddf1.distinct('col_1')
        """
        from .functions.etl.distinct import distinct

        settings = {'columns': cols}

        def task_distinct(df, params):
            return distinct(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'distinct',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_distinct, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def drop(self, columns):
        """
        Remove some columns from DDF.

        Is it a Lazy function: Yes

        :param columns: A list of columns names to be removed;
        :return: DDF

        :Example:

        >>> ddf1.drop(['col_1', 'col_2'])
        """

        settings = {'columns': columns}

        from .functions.etl.drop import drop

        def task_drop(df, params):
            return drop(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'drop',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_drop, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def drop_duplicates(self, cols):
        """
        Alias for distinct.

        :Example:

        >>> ddf1.drop_duplicates('col_1')
        """
        return self.distinct(cols)

    def dropna(self, subset=None, mode='REMOVE_ROW', how='any', thresh=None):
        """
        Cleans missing rows or columns fields.

        Is it a Lazy function: Yes, if mode is *"REMOVE_ROW"*, otherwise is No

        :param subset: A list of attributes to evaluate;
        :param mode: *"REMOVE_ROW"** to remove entire row (default) or
         **"REMOVE_COLUMN"** to remove a column.
        :param thresh: int, default None If specified, drop rows that have less
          than thresh non-null values. This overwrites the how parameter.
        :param how: 'any' or 'all'. If 'any', drop a row if it contains any
          nulls. If 'all', drop a row only if all its values are null.
        :return: DDF

        :Example:

        >>> ddf1.dropna(['col_1'], mode='REMOVE_ROW')
        """

        from .functions.etl.clean_missing import DropNaN

        if mode is 'REMOVE_ROW':
            lazy = self.OPT_SERIAL

            def task_dropna(df, _):
                return DropNaN(subset, how=how, thresh=thresh, mode=mode)\
                    .drop_rows(df)

        else:
            lazy = self.OPT_OTHER

            def task_dropna(df, _):
                return DropNaN(subset, how=how, thresh=thresh, mode=mode)\
                    .preprocessing(df).drop_columns(df)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'dropna',
             'status': 'WAIT',
             'optimization': lazy,
             'function': [task_dropna, {}],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }
        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def explode(self, column):
        """
        Returns a new row for each element in the given array.

        Is it a Lazy function: Yes

        :param column: Column name to be unnest;
        :return: DDF

        :Example:

        >>> ddf1.explode('col_1')
        """

        settings = {'column': column}

        from .functions.etl.explode import explode

        def task_explode(df, params):
            return explode(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'explode',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_explode, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def filter(self, expr):
        """
        Filters rows using the given condition.

        Is it a Lazy function: Yes

        :param expr: A filtering function;
        :return: DDF

        .. seealso:: Visit this `link <https://pandas.pydata.org/pandas-docs/
         stable/generated/pandas.DataFrame.query.html>`__ to more
         information about query options.

        :Example:

        >>> ddf1.filter("(col_1 == 'male') and (col_3 > 42)")
        """

        from .functions.etl.filter import filter_rows
        settings = {'query': expr}

        def task_filter(df, params):
            return filter_rows(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'filter',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_filter, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def geo_within(self, shp_object, lat_col, lon_col, polygon,
                   attributes=None, suffix='_shp'):
        """
        Returns the sectors that the each point belongs.

        Is it a Lazy function: No

        :param shp_object: The DDF with the shapefile information;
        :param lat_col: Column which represents the Latitute field in the data;
        :param lon_col: Column which represents the Longitude field in the data;
        :param polygon: Field in shp_object where is store the
            coordinates of each sector;
        :param attributes: Attributes list to retrieve from shapefile,
            empty to all (default, None);
        :param suffix: Shapefile attributes suffix (default, *'_shp'*);
        :return: DDF

        :Example:

        >>> ddf2.geo_within(ddf1, 'LATITUDE', 'LONGITUDE', 'points')
        """

        from .functions.geo import GeoWithinOperation

        settings = dict()
        settings['lat_col'] = lat_col
        settings['lon_col'] = lon_col
        if attributes is not None:
            settings['attributes'] = attributes
        settings['polygon'] = polygon
        settings['alias'] = suffix

        def task_geo_within(df, params):
            return GeoWithinOperation().transform(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'geo_within',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_geo_within, settings],
             'parent': [self.last_uuid, shp_object.last_uuid],
             'output': 1,
             'input': 2
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, shp_object.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + shp_object.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def hash_partition(self, columns, nfrag=None):
        """
        Hash partitioning is a partitioning technique where data
        is stored separately in different fragments by a hash function.

        Is it a Lazy function: No

        :param columns: Columns to be used as key in a hash function;
        :param nfrag: Number of fragments (default, keep the input nfrag).

        :Example:

        >>> ddf2 = ddf1.hash_partition(columns=['col1', col2])
        """

        from .functions.etl.hash_partitioner import hash_partition

        if not isinstance(columns, list):
            columns = [columns]

        settings = {'columns': columns}

        if nfrag is not None:
            settings['nfrag'] = nfrag

        def task_hash_partition(df, params):
            return hash_partition(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'hash_partition',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_hash_partition, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def intersect(self, data2):
        """
        Returns a new DDF containing rows in both DDF. This is equivalent to
        INTERSECT in SQL.

        Is it a Lazy function: No

        :param data2: DDF
        :return: DDF

        :Example:

        >>> ddf2.intersect(ddf1)
        """

        from .functions.etl.intersect import intersect

        settings = {'distinct': True}

        def task_intersect(df, params):
            return intersect(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'intersect',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_intersect, settings],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2,
             'info': True,
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def intersect_all(self, data2):
        """
        Returns a new DDF containing rows in both DDF while preserving
        duplicates. This is equivalent to INTERSECT ALL in SQL.

        Is it a Lazy function: No

        :param data2: DDF
        :return: DDF

        :Example:

        >>> ddf2.intersect_all(ddf1)
        """

        from .functions.etl.intersect import intersect

        settings = {'distinct': False}

        def task_intersect(df, params):
            return intersect(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'intersect',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_intersect, settings],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2,
             'info': True,
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def join(self, data2, key1=None, key2=None, mode='inner',
             suffixes=None, keep_keys=False, case=True):
        """
        Joins two DDF using the given join expression.

        Is it a Lazy function: No

        :param data2: Second DDF;
        :param key1: List of keys of first DDF;
        :param key2: List of keys of second DDF;
        :param mode: How to handle the operation of the two objects. {‘left’,
         ‘right’, ‘inner’}, default ‘inner’
        :param suffixes: A list of suffix to be used in overlapping columns.
         Default is ['_l', '_r'];
        :param keep_keys: True to keep keys of second DDF (default is False);
        :param case: True to keep the keys as case sensitive;
        :return: DDF

        :Example:

        >>> ddf1.join(ddf2, key1=['col_1'], key2=['col_1'], mode='inner')
        """

        if key1 is None:
            key1 = []

        if key2 is None:
            key2 = []

        if suffixes is None:
            suffixes = ['_l', '_r']

        from .functions.etl.join import JoinOperation

        settings = {'key1': key1,
                    'key2': key2,
                    'option': mode,
                    'keep_keys': keep_keys,
                    'case': case,
                    'suffixes': suffixes}

        def task_join(df, params):
            return JoinOperation().transform(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'join',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_join, settings],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2,
             'info': True}

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def kolmogorov_smirnov_one_sample(self, col, distribution='norm',
                                      mode='asymp', args=None):
        """
        Perform the Kolmogorov-Smirnov test for goodness of fit. This
        implementation of Kolmogorov–Smirnov test is a two-sided test
        for the null hypothesis that the sample is drawn from a continuous
        distribution.

        Is it a Lazy function: No

         :param col: sample column name;
         :param distribution: Name of distribution (default is 'norm');
         :param mode: Defines the distribution used for calculating the p-value.
            - 'approx' : use approximation to exact distribution
            - 'asymp' : use asymptotic distribution of test statistic
        :param args: A tuple of distribution parameters. Default is (0,1);
        :return: KS statistic and two-tailed p-value

        .. seealso:: Visit this `link <https://docs.scipy.org/doc/scipy-0.14.0/
         reference/stats.html#module-scipy.stats>`__ to see all supported
         distributions.

        .. note:: The KS statistic is the absolute max distance (supremum)
         between the CDFs of the two samples. The closer this number is to
         0 the more likely it is that the two samples were drawn from the
         same distribution.

         The p-value returned by the KS test has the same interpretation
         as other p-values. You reject the null hypothesis that the two
         samples were drawn from the same distribution if the p-value is
         less than your significance level.

        :Example:

        >>> ddf1.kolmogorov_smirnov_one_sample(col='col_1')
        """
        from ddf_library.functions.statistics.kolmogorov_smirnov \
            import kolmogorov_smirnov_one_sample

        settings = {'col': col, 'distribution': distribution, 'mode': mode}
        if args is not None:
            settings['args'] = args

        df, nfrag, tmp = self._ddf_initial_setup(self)

        result = kolmogorov_smirnov_one_sample(df, settings)

        return result

    def map(self, f, alias):
        """
        Apply a function to each row of this DDF.

        Is it a Lazy function: Yes

        :param f: Lambda function that will take each element of this data
         set as a parameter;
        :param alias: name of column to put the result;
        :return: DDF

        :Example:

        >>> ddf1.map(lambda row: row['col_0'].split(','), 'col_0_new')
        """

        settings = {'function': f, 'alias': alias}

        from .functions.etl.map import map as task

        def task_map(df, params):
            return task(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'map',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_map, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1}

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def range_partition(self, columns, ascending=None, nfrag=None):
        """
        Range partitioning is a partitioning technique where ranges of data
        is stored separately in different fragments.

        Is it a Lazy function: No

        :param columns: Columns to be used as key;
        :param ascending: Order of each key (True to ascending order);
        :param nfrag: Number of fragments (default, keep the input nfrag).

        :Example:

        >>> ddf2 = ddf1.range_partition(columns=['col1', col2],
        >>>                             ascending=[True, False])
        """

        from .functions.etl.range_partitioner import range_partition

        if not isinstance(columns, list):
            columns = [columns]

        if ascending is None:
            ascending = True

        if not isinstance(ascending, list):
            ascending = [ascending for _ in columns]

        settings = {'columns': columns, 'ascending': ascending}

        if nfrag is not None:
            settings['nfrag'] = nfrag

        def task_range_partition(df, params):
            return range_partition(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'range_partition',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_range_partition, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def repartition(self, nfrag=-1, distribution=None):
        """

        """

        from .functions.etl.repartition import repartition

        settings = {'nfrag': nfrag}

        if distribution is not None:
            settings['distribution'] = distribution

        def task_repartition(df, params):
            return repartition(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'repartition',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_repartition, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def replace(self, replaces, subset=None):
        """
        Replace one or more values to new ones.

        Is it a Lazy function: Yes

        :param replaces: dict-like `to_replace`;
        :param subset: A list of columns to be applied (default is
         None to applies in all columns);
        :return: DDF

        :Example:

        >>> ddf1.replace({0: 'No', 1: 'Yes'}, subset='col_1')
        """

        from .functions.etl.replace_values import replace_value, preprocessing

        settings = {'replaces': replaces}
        if subset is not None:
            settings['subset'] = subset

        settings = preprocessing(settings)

        def task_replace(df, params):
            return replace_value(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'replace',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_replace, settings],
             'parent': [self.last_uuid],
             'output': 1, 'input': 1}

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def sample(self, value=None, seed=None):
        """
        Returns a sampled subset.

        Is it a Lazy function: No

        :param value: None to sample a random amount of records (default),
            a integer or float N to sample a N random records;
        :param seed: optional, seed for the random operation.
        :return: DDF

        :Example:

        >>> ddf1.sample(10)  # to sample 10 rows
        >>> ddf1.sample(0.5) # to sample half of the elements
        >>> ddf1.sample()  # a random sample
        """

        from .functions.etl.sample import sample
        settings = dict()
        settings['seed'] = seed

        if value:
            """Sample a N random records"""
            settings['type'] = 'value'
            settings['value'] = value

        else:
            """Sample a random amount of records"""
            settings['type'] = 'percent'

        def task_sample(df, params):
            return sample(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sample',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_sample, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def save(self, filename, filetype='csv', storage='hdfs',
             header=True, mode='overwrite'):
        """
        Save the data in the storage.

        Is it a Lazy function: Yes

        :param filename: output name;
        :param filetype: format file, CSV or JSON;
        :param storage: 'fs' to commom file system or 'hdfs' to use HDFS;
        :param header: save with the columns header;
        :param mode: 'overwrite' if file exists, 'ignore' or 'error'
        :return:
        """

        from ddf_library.functions.etl.save_data import SaveOperation

        settings = dict()
        settings['filename'] = filename
        settings['format'] = filetype
        settings['storage'] = storage
        settings['header'] = header
        settings['mode'] = mode

        settings = SaveOperation().preprocessing(settings, len(self.partitions))

        def task_save(df, params):
            return SaveOperation().transform_serial(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'save',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_save, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def schema(self):

        info = self._get_info()
        tmp = pd.DataFrame.from_dict({'columns': info['cols'],
                                      'dtypes': info['dtypes']})
        return tmp

    def select(self, columns):
        """
        Projects a set of expressions and returns a new DDF.

        Is it a Lazy function: Yes

        :param columns: list of column names (string);
        :return: DDF

        :Example:

        >>> ddf1.select(['col_1', 'col_2'])
        """

        from .functions.etl.select import select

        settings = {'columns': columns}

        def task_select(df, params):
            return select(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'select',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_select, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def select_expression(self, *exprs):
        """
        Projects a set of SQL expressions and returns a new DataFrame.
        This is a variant of select() that accepts SQL expressions.

        Is it a Lazy function: Yes

        :param exprs: SQL expressions.
        :return: DDF

        .. note: These operations are supported by select_exprs:

        * Arithmetic operations except for the left shift (<<) and right shift
         (>>) operators, e.g., 'col' + 2 * pi / s ** 4 % 42 - the_golden_ratio
        * Comparison operations, including chained comparisons,
         e.g., 2 < df < df2
        * Boolean operations, e.g., df < df2 and df3 < df4 or not df_bool
        * list and tuple literals, e.g., [1, 2] or (1, 2)
        * Subscript expressions, e.g., df[0]
        * Math functions: sin, cos, exp, log, expm1, log1p, sqrt, sinh, cosh,
         tanh, arcsin, arccos, arctan, arccosh, arcsinh, arctanh, abs, arctan2
         and log10.
        * This Python syntax is not allowed:

         * Expressions

          - Function calls other than math functions.
          - is/is not operations
          - if expressions
          - lambda expressions
          - list/set/dict comprehensions
          - Literal dict and set expressions
          - yield expressions
          - Generator expressions
          - Boolean expressions consisting of only scalar values

         * Statements: Neither simple nor compound statements are allowed.
          This includes things like for, while, and if.

        You must explicitly reference any local variable that you want to use
        in an expression by placing the @ character in front of the name.

        .. seealso:: Visit this `link <https://pandas-docs.github.io/pandas-docs
            -travis/reference/api/pandas.eval.html#pandas.eval>`__ to more
            information about eval options.

        :Example:

        >>> ddf1.select_exprs('col1 = age * 2', "abs(age)")
        """

        from .functions.etl.select import select_exprs

        settings = {'exprs': exprs}

        def task_select_exprs(df, params):
            return select_exprs(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'select_exprs',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_select_exprs, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def show(self, n=20):
        """
        Print the DDF contents in a concatenated pandas's DataFrame.

        :param n: A number of rows in the result (default is 20);
        :return: DataFrame in stdout

        :Example:

        >>> ddf1.show()
        """

        self._check_cache()

        n_rows_frags = self.count_rows(False)

        from .functions.etl.take import take
        res = take(self.partitions, {'value': n,
                                     'info': [{'size': n_rows_frags}]})
        res = compss_wait_on(res['data'])

        df = concatenate_pandas(res)

        print(df)
        return self

    def sort(self, cols,  ascending=None):
        """
        Returns a sorted DDF by the specified column(s).

        Is it a Lazy function: No

        :param cols: list of columns to be sorted;
        :param ascending: list indicating whether the sort order
            is ascending (True) for each column (Default, True);
        :return: DDF

        :Example:

        >>> dd1.sort(['col_1', 'col_2'], ascending=[True, False])
        """
        mode = 'by_range'
        from .functions.etl.sort import SortOperation

        settings = {'columns': cols, 'ascending': ascending, 'algorithm': mode}

        def task_sort(df, params):
            return SortOperation().transform(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sort',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_sort, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True  # only to forward
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def split(self, percentage=0.5, seed=None):
        """
        Randomly splits a DDF into two DDF.

        Is it a Lazy function: No

        :param percentage: percentage to split the data (default, 0.5);
        :param seed: optional, seed in case of deterministic random operation;
        :return: DDF

        :Example:

        >>> ddf2a, ddf2b = ddf1.split(0.5)
        """

        from .functions.etl.split import random_split

        settings = {'percentage': percentage, 'seed': seed}

        def task_split(df, params):
            return random_split(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'split',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_split, settings],
             'parent': [self.last_uuid],
             'output': 2,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list,
                   last_uuid=new_state_uuid, settings={'input': 0}).cache(), \
            DDF(task_list=self.task_list, last_uuid=new_state_uuid,
                settings={'input': 1}).cache()

    def take(self, num):
        """
        Returns the first num rows.

        Is it a Lazy function: No

        :param num: number of rows to retrieve;
        :return: DDF

        :Example:

        >>> ddf1.take(10)
        """

        from .functions.etl.take import take
        settings = {'value': num}

        def task_take(df, params):
            return take(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'take',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_take, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def to_df(self, columns=None, split=False):
        """
        Returns the DDF contents as a pandas's DataFrame.

        :param columns: Optional, list of new column names (string);
        :param split: True to keep data in partitions (default, False);
        :return: Pandas's DataFrame

        :Example:

        >>> df = ddf1.to_df(['col_1', 'col_2'])
        """
        last_last_uuid = self.task_list[-2]
        self._check_cache()

        res = compss_wait_on(self.partitions)
        n_input = self.settings['input']

        COMPSsContext.tasks_map[self.last_uuid]['function'][n_input] = res

        if len(self.task_list) > 2:
            COMPSsContext.tasks_map[last_last_uuid]['function'][n_input] = res

        if isinstance(columns, str):
            columns = [columns]

        if split:
            if columns:
                df = [d[columns] for d in res]
            else:
                df = res
        else:
            df = concatenate_pandas(res)
            if columns:
                df = df[columns]

            df.reset_index(drop=True, inplace=True)
        return df

    def union(self, data2):
        """
        Combine this data set with some other DDF. Also as standard in SQL,
        this function resolves columns by position (not by name). The old names
        are replaced to 'col_' + index.

        Is it a Lazy function: No

        :param data2:
        :return: DDF

        :Example:

        >>> ddf1.union(ddf2)
        """

        from .functions.etl.union import union

        settings = {'by_name': False}

        def task_union(df, params):
            return union(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'union',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_union, settings],
             'parent': [self.last_uuid,  data2.last_uuid],
             'output': 1,
             'input': 2,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def union_by_name(self, data2):
        """
        Combine this data set with some other DDF. This function resolves
         columns by name (not by position).

        Is it a Lazy function: No

        :param data2:
        :return: DDF

        :Example:

        >>> ddf1.union_by_name(ddf2)
        """

        from .functions.etl.union import union

        settings = {'by_name': True}

        def task_union(df, params):
            return union(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'union',
             'status': 'WAIT',
             'optimization': self.OPT_OTHER,
             'function': [task_union, settings],
             'parent': [self.last_uuid,  data2.last_uuid],
             'output': 1,
             'input': 2,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def rename(self, old_column, new_column):
        """
        Returns a new DDF by renaming an existing column. This is a no-op if
        schema doesn’t contain the given column name.

        Is it a Lazy function: Yes

        :param old_column: String or list of strings with columns to rename;
        :param new_column: String or list of strings with new names.

        :return: DDF
        """

        from .functions.etl.attributes_changer import with_column_renamed

        if not isinstance(old_column, list):
            old_column = [old_column]

        if not isinstance(new_column, list):
            new_column = [new_column]

        settings = {'old_column': old_column, 'new_column': new_column}

        def task_rename(df, params):
            return with_column_renamed(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'rename',
             'status': 'WAIT',
             'optimization': self.OPT_SERIAL,
             'function': [task_rename, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])

        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)
