#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.

Public classes:

  - :class:`DDF`:
      Distributed DataFrame (DDF), the abstraction of this library.
"""

import uuid

import pandas as pd
import numpy as np

from pycompss.api.api import compss_wait_on
from pycompss.functions.reduce import merge_reduce


from collections import OrderedDict

import copy

import context


class DDFSketch(object):

    """
    Basic functions that are necessary when submit a new operation
    """

    def __init__(self):
        pass

    @staticmethod
    def _merge_tasks_list(seq):
        """
        Merge two list of tasks removing duplicated tasks

        :param seq: list with possible duplicated elements
        :return:
        """
        seen = set()
        return [x for x in seq if x not in seen and not seen.add(x)]

    @staticmethod
    def _generate_uuid():
        """
        Generate a unique id
        :return: uuid
        """
        new_state_uuid = str(uuid.uuid4())
        while new_state_uuid in context.COMPSsContext.tasks_map:
            new_state_uuid = str(uuid.uuid4())
        return new_state_uuid

    @staticmethod
    def _set_n_input(state_uuid, idx):
        """
        Method to inform the index of the input data

        :param state_uuid: id of the current task
        :param idx: idx of input data
        :return:
        """

        if 'n_input' not in context.COMPSsContext.tasks_map[state_uuid]:
            context.COMPSsContext.tasks_map[state_uuid]['n_input'] = []
        context.COMPSsContext.tasks_map[state_uuid]['n_input'].append(idx)

    @staticmethod
    def _ddf_inital_setup(data):
        tmp = data.cache()
        n_input = context.COMPSsContext.tasks_map[tmp.last_uuid]['n_input'][0]
        if n_input == -1:
            n_input = 0
        df = context.COMPSsContext.tasks_map[tmp.last_uuid]['function'][n_input]
        nfrag = len(df)
        return df, nfrag, tmp

    def _ddf_add_task(self, task_name, status, lazy, function,
                      parent, n_output, n_input, info=None):

        uuid_key = self._generate_uuid()
        context.COMPSsContext.tasks_map[uuid_key] = {
            'name': task_name,
            'status': status,
            'lazy': lazy,
            'function': function,
            'parent': parent,
            'output': n_output,
            'input': n_input
        }

        if info:
            info = merge_reduce(context.merge_schema, info)
            context.COMPSsContext.schemas_map[uuid_key] = {0: info}
        return uuid_key


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

        # self._schema = context.COMPSsContext.schemas_map.get(last_uuid, list())

        self.opt = OrderedDict()
        self.partial_sizes = list()
        self.partitions = list()

        if last_uuid != 'init':

            self.task_list = copy.deepcopy(task_list)
            self.task_list.append(last_uuid)

        else:
            last_uuid = str(uuid.uuid4())

            self.task_list = list()
            self.task_list.append(last_uuid)

            context.COMPSsContext.tasks_map[last_uuid] = \
                {'name': 'init',
                 'lazy': False,
                 'input': 0,
                 'parent': [],
                 'status': 'COMPLETED'
                 }

        self.last_uuid = last_uuid

    def _execute_task(self, f):
        """
        Used to execute all non-lazy functions.

        :param f: a list that contains the current task and its parameters.
        :return:
        """

        function, settings = f
        if len(self.partitions) > 1:
            partitions = [self.partitions[k] for k in self.partitions]
        else:
            partitions = self.partitions[0]

        output = function(partitions, settings)
        return output

    def _execute_lazy(self, opt):

        """
        Used to execute a group of lazy tasks. This method submit
        multiple 'context.task_bundle', one for each data fragment.

        :param opt: sequence of functions and parameters to be executed in
            each fragment
        :return:
        """

        data = self.partitions
        tmp = []
        if isinstance(data, dict):
            if len(data) > 1:
                for k in data:
                    tmp.append(data[k])
            else:
                for k in data:
                    tmp = data[k]
        else:
            tmp = data

        partitions = [[] for _ in tmp]
        info = [[] for _ in tmp]
        for idfrag, p in enumerate(tmp):
            partitions[idfrag], info[idfrag] = context.task_bundle(p, opt, idfrag)

        info = merge_reduce(context.merge_schema, info)
        return partitions, info

    def load_text(self, filename, num_of_parts=4, header=True,
                  sep=',', storage='hdfs', host='localhost', port=9000,
                  distributed=False, dtype='str'):
        """
        Create a DDF from a commom file system or from HDFS.

        :param filename: Input file name;
        :param num_of_parts: number of partitions (default, 4);
        :param header: Use the first line as DataFrame header (default, True);
        :param dtype: Type name or dict of column (default, 'str'). Data type
         for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32, ‘c’:
         ‘Int64’} Use str or object together with suitable na_values settings
         to preserve and not interpret dtype;
        :param sep: separator delimiter (default, ',');
        :param storage: *'hdfs'* to use HDFS as storage or *'fs'* to use the
         common file sytem;
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

        from functions.etl.read_data import DataReader

        data_reader = DataReader(filename, nfrag=num_of_parts,
                                 format='csv', storage=storage,
                                 distributed=distributed,
                                 dtype=dtype, separator=sep, header=header,
                                 na_values='', host=host, port=port)

        if storage is 'fs':

            result, info = data_reader.transform()

            new_state_uuid = self._generate_uuid()
            context.COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'load_text',
                 'status': 'COMPLETED',
                 'lazy': False,
                 'function': {0: result},
                 'output': 1,
                 'input': 0,
                 'parent': [self.last_uuid]
                 }

            context.COMPSsContext.schemas_map[new_state_uuid] = {0: info}
        else:
            blocks = data_reader.get_blocks()

            context.COMPSsContext.tasks_map[self.last_uuid]['function'] = \
                {0: blocks}

            new_state_uuid = self._generate_uuid()
            context.COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'load_text',
                 'status': 'WAIT',
                 'lazy': True,
                 'function': [data_reader.read_hdfs_serial, {}],
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

        from functions.etl.parallelize import parallelize

        result, info = parallelize(df, num_of_parts)

        info = merge_reduce(context.merge_schema, info)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'parallelize',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: result},
             'output': 1, 'input': 0,
             'parent': [self.last_uuid]
             }

        context.COMPSsContext.schemas_map[new_state_uuid] = {0: info}

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def load_shapefile(self, shp_path, dbf_path, polygon='points',
                       attributes=[], num_of_parts=4):
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

        settings = dict()
        settings['shp_path'] = shp_path
        settings['dbf_path'] = dbf_path
        settings['polygon'] = polygon
        settings['attributes'] = attributes

        from functions.geo import read_shapefile

        result, info = read_shapefile(settings, num_of_parts)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'load_shapefile',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: result},
             'output': 1, 'input': 0,
             'parent': [self.last_uuid]
             }
        context.COMPSsContext.schemas_map[new_state_uuid] = {0: info}

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def _collect(self):
        """
        #TODO Check it
        :return:
        """

        if context.COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
            self.partitions = \
                context.COMPSsContext.tasks_map[self.last_uuid]['function'][0]

        self.partitions = compss_wait_on(self.partitions)

    def cache(self):
        """
        Compute all tasks until the current state

        :return: DDF

        :Example:

        >>> ddf1.cache()
        """

        # TODO: no momento, é necessario para lidar com split
        # if context.COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
        #     print 'cache skipped'
        #     return self

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sync',
             'status': 'WAIT',
             'lazy': False,
             'parent': [self.last_uuid],
             'function': [[]]
             }

        self._set_n_input(new_state_uuid, self.settings['input'])

        tmp = DDF(task_list=self.task_list,
                  last_uuid=new_state_uuid,
                  settings=self.settings)._run_compss_context(new_state_uuid)

        return tmp

    def _check_cache(self):

        cached = False

        for _ in range(2):
            if context.COMPSsContext.tasks_map[self.last_uuid][
                'status'] == 'COMPLETED':
                n_input = \
                    context.COMPSsContext.tasks_map[self.last_uuid]['n_input'][
                        0]
                # if n_input == -1:
                #     n_input = 0
                self.partitions = \
                    context.COMPSsContext.tasks_map[self.last_uuid]['function'][
                        n_input]
                cached = True
                break
            else:
                self.cache()

        if not cached:
            raise Exception("ERROR - toPandas - not cached")

    def _run_compss_context(self, wanted=None):
        context.COMPSsContext().run_workflow(wanted)
        return self

    # TODO: CHECK it
    def num_of_partitions(self):
        """
        Returns the number of data partitions (Task parallelism).

        :return: integer

        :Example:

        >>> print ddf1.num_of_partitions()
        """
        size = len(context.COMPSsContext().tasks_map[self.last_uuid]['function'])
        return size

    def count(self):
        """
        Return a number of rows in this DDF.

        :return: integer

        :Example:

        >>> print ddf1.count()
        """

        #TODO: será q n preciso nos outros?
        last_last_uuid = self.task_list[-2]
        cached = False

        for _ in range(2):
            if context.COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
                n_input = context.COMPSsContext.tasks_map[self.last_uuid]['n_input'][0]
                if n_input == -1:
                    n_input = 0
                self.partitions = \
                    context.COMPSsContext.tasks_map[self.last_uuid]['function'][n_input]
                cached = True
                break
            else:
                self.cache()

        if not cached:
            raise Exception("ERROR - toPandas - not cached")

        info = self._get_info()

        size = sum(info[2])
        #
        # def task_count(df, params):
        #     return len(df)
        #
        # new_state_uuid = self._generate_uuid()
        # context.COMPSsContext.tasks_map[new_state_uuid] = \
        #     {'name': 'count',
        #      'status': 'WAIT',
        #      'lazy': True,
        #      'function': [task_count, {}],
        #      'parent': [self.last_uuid],
        #      'output': 1,
        #      'input': 1
        #      }
        # self._set_n_input(new_state_uuid, self.settings['input'])
        #
        # tmp = DDF(task_list=self.task_list, last_uuid=new_state_uuid).cache()
        #
        # result = context.COMPSsContext.tasks_map[new_state_uuid]['function']
        # res = compss_wait_on(result[0])
        # del tmp
        # res = sum(res)

        return size

    def with_column(self, old_column, new_column=None, cast=None):
        """
        Rename or change the data's type of some columns.

        Is it a Lazy function: Yes

        :param old_column: the current column name;
        :param new_column: the new column name;
        :param cast: 'keep' (if None), 'integer', 'string', 'double', 'Date',
            'Date/time';
        :return: DDF
        """

        from functions.etl.attributes_changer import \
            attributes_changer, preprocessing

        if not isinstance(old_column, list):
            old_column = [old_column]

        if not isinstance(new_column, list):
            new_column = [new_column]

        if not isinstance(cast, list):
            cast = [cast]

        settings = dict()
        settings['attributes'] = old_column
        settings['new_name'] = new_column
        settings['new_data_type'] = cast

        settings = preprocessing(settings)

        def task_with_column(df, params):
            return attributes_changer(df, params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'with_column',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_with_column, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])

        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def add_column(self, data2, suffixes=["_l", "_r"]):
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
        from functions.etl.add_columns import AddColumnsOperation

        def task_add_column(df, params):
            return AddColumnsOperation().transform(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'add_column',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_add_column, suffixes],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def aggregation(self, group_by, exprs, aliases):
        """
        Computes aggregates and returns the result as a DDF.

        Is it a Lazy function: No

        :param group_by: A list of columns to be grouped;
        :param exprs: A dict of lists with all operations in some column;
        :param aliases: A dict of lists with all new names;
        :return: DFF

        :Example:

        >>> ddf1.aggregation(group_by=['col_1'], exprs={'col_1': ['count']},
        >>>                  aliases={'col_1': ["new_col"]})
        """

        settings = dict()
        settings['columns'] = group_by
        settings['operation'] = exprs
        settings['aliases'] = aliases

        from functions.etl.aggregation import AggregationOperation

        def task_aggregation(df, params):
            return AggregationOperation().transform(df, params, len(df))

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'aggregation',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_aggregation, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def clean_missing(self, columns, mode='REMOVE_ROW', value=None):
        """
        Cleans missing rows or columns fields.

        Is it a Lazy function: Yes, if mode is *"VALUE"* or *"REMOVE_ROW"*,
        otherwise is No

        :param columns: A list of attributes to evaluate;
        :param mode: action in case of missing values: *"VALUE"* to replace by
         parameter "value"; *"REMOVE_ROW"** to remove entire row (default);
         *"MEDIAN"* to replace by median value;  *"MODE"* to replace by
         mode value; *"MEAN"* to replace by mean value *"REMOVE_COLUMN"*
         to remove entire column;
        :param value: Value to be replaced (only if mode is *"VALUE"*)
        :return: DDF

        :Example:

        >>> ddf1.clean_missing(['col_1'], mode='REMOVE_ROW')
        """

        from functions.etl.clean_missing import CleanMissingOperation

        settings = dict()
        settings['attributes'] = columns
        settings['cleaning_mode'] = mode
        settings['value'] = value

        lazy = False
        if mode in ['VALUE', 'REMOVE_ROW']:
            lazy = True

            def task_clean_missing(df, params):
                return CleanMissingOperation().transform_serial(df, params)

        else:
            def task_clean_missing(df, params):
                return CleanMissingOperation().transform(df, params, len(df))

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'clean_missing',
             'status': 'WAIT',
             'lazy': lazy,
             'function': [task_clean_missing, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }
        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def columns(self):

        columns = self._get_info()[0]
        return columns

    def cross_join(self, data2):
        """
        Returns the cartesian product with another DDF.

        :param data2: Right side of the cartesian product;
        :return: DDF.

        :Example:

        >>> ddf1.cross_join(ddf2)
        """
        from functions.etl.cross_join import crossjoin

        def task_cross_join(df, params):
            return crossjoin(df[0], df[1])

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'cross_join',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_cross_join, {}],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

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

        df, nfrag, tmp = self._ddf_inital_setup(self)

        from functions.etl.describe import describe

        if not columns:
            columns = []

        result = describe(df, columns)

        return result

    def difference(self, data2):
        """
        Returns a new DDF with containing rows in the first frame but not
        in the second one.

        Is it a Lazy function: No

        :param data2: second DDF;
        :return: DDF

        :Example:

        >>> ddf1.difference(ddf2)
        """
        from functions.etl.difference import difference

        def task_difference(df, params):
            return difference(df[0], df[1])

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'difference',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_difference, {}],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2
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

        >>> ddf1.distinct(['col_1'])
        """
        from functions.etl.distinct import distinct

        def task_distinct(df, params):
            return distinct(df, params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'distinct',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_distinct, cols],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
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

        from functions.etl.drop import drop

        def task_drop(df, cols):
            return drop(df, cols)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'drop',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_drop, columns],
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

        from functions.etl.filter import filter

        def task_filter(df, query):
            return filter(df, query)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'filter',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_filter, expr],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def geo_within(self, shp_object, lat_col, lon_col, polygon,
                   attributes=[], suffix='_shp'):
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

        from functions.geo import GeoWithinOperation

        settings = dict()
        settings['lat_col'] = lat_col
        settings['lon_col'] = lon_col
        settings['attributes'] = attributes
        settings['polygon'] = polygon
        settings['alias'] = suffix

        def task_geo_within(df, params):
            return GeoWithinOperation().transform(df[0], df[1], params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'geo_within',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_geo_within, settings],
             'parent': [self.last_uuid, shp_object.last_uuid],
             'output': 1,
             'input': 2
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, shp_object.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + shp_object.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def intersect(self, data2):
        """
        Returns a new DDF containing rows in both DDF.

        Is it a Lazy function: No

        :param data2: DDF
        :return: DDF

        :Example:

        >>> ddf2.intersect(ddf1)
        """

        from functions.etl.intersect import IntersectionOperation

        def task_intersect(df, params):
            return IntersectionOperation().transform(df[0], df[1])

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'intersect',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_intersect, {}],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

    def join(self, data2, key1=[], key2=[], mode='inner',
             suffixes=['_l', '_r'], keep_keys=False,
             case=True, sort=True):
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
        :param sort: To sort data at first;
        :return: DDF

        :Example:

        >>> ddf1.join(ddf2, key1=['col_1'], key2=['col_1'], mode='inner')
        """

        from functions.etl.join import JoinOperation

        settings = dict()
        settings['key1'] = key1
        settings['key2'] = key2
        settings['option'] = mode
        settings['keep_keys'] = keep_keys
        settings['case'] = case
        settings['sort'] = sort
        settings['suffixes'] = suffixes

        def task_join(df, params):
            return JoinOperation().transform(df[0], df[1], params, len(df[0]))

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'join',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_join, settings],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1, 'input': 2}

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

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

        from functions.etl.map import map as task

        def task_map(df, params):
            return task(df, params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'map',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_map, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1}

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

        from functions.etl.replace_values import replace_value, preprocessing

        settings = {'replaces': replaces, 'subset': subset}
        settings = preprocessing(settings)

        def task_replace(df, params):
            return replace_value(df, params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'replace',
             'status': 'WAIT',
             'lazy': True,
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

        from functions.etl.sample import sample
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
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sample',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_sample, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def save(self, filename, format='csv', storage='hdfs',
             header=True, mode='overwrite'):
        """
        Save the data in the storage.

        Is it a Lazy function: Yes

        :param filename: output name;
        :param format: format file, CSV or JSON;
        :param storage: 'fs' to commom file system or 'hdfs' to use HDFS;
        :param header: save with the columns header;
        :param mode: 'overwrite' if file exists, 'ignore' or 'error'
        :return:
        """

        from ddf.functions.etl.save_data import SaveOperation

        settings = dict()
        settings['filename'] = filename
        settings['format'] = format
        settings['storage'] = storage
        settings['header'] = header
        settings['mode'] = mode

        settings = SaveOperation().preprocessing(settings, len(self.partitions))

        def task_save(df, params):
            return SaveOperation().transform_serial(df, params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'save',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_save, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def _get_info(self):

        self._check_cache()
        n_input = self.settings['input']
        info = context.COMPSsContext.schemas_map[self.last_uuid][n_input]
        info = compss_wait_on(info)

        context.COMPSsContext.schemas_map[self.last_uuid][n_input] = info
        return info

    def schema(self):

        info = self._get_info()
        tmp = pd.DataFrame.from_dict({'columns': info[0], 'dtype': info[1]})
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

        from functions.etl.select import select

        def task_select(df, fields):
            return select(df, fields)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'select',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_select, columns],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)


    def show(self, n=20):
        """
        Returns the DDF contents in a concatenated pandas's DataFrame.

        :param n: A number of rows in the result (default is 20);
        :return: Pandas's DataFrame

        :Example:

        >>> df = ddf1.show()
        """
        last_last_uuid = self.task_list[-2]
        self._check_cache()
        # cached = False
        #
        # for _ in range(2):
        #     if context.COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
        #         n_input = context.COMPSsContext.tasks_map[self.last_uuid]['n_input'][0]
        #         if n_input == -1:
        #             n_input = 0
        #         self.partitions = \
        #             context.COMPSsContext.tasks_map[self.last_uuid]['function'][n_input]
        #         cached = True
        #         break
        #     else:
        #         self.cache()
        #
        # if not cached:
        #     raise Exception("ERROR - toPandas - not cached")

        res = compss_wait_on(self.partitions)
        n_input = self.settings['input']
        # if n_input == -1:
        #     n_input = 0

        context.COMPSsContext.tasks_map[self.last_uuid]['function'][n_input] = res
        if len(self.task_list) > 2:
            context.COMPSsContext.tasks_map[last_last_uuid]['function'][n_input] = res

        df = pd.concat(res, sort=True)[:abs(n)]
        df.reset_index(drop=True, inplace=True)
        return df

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

        from functions.etl.sort import SortOperation

        settings = {'columns': cols, 'ascending': ascending}

        def task_sort(df, params):
            return SortOperation().transform(df, params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sort',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_sort, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
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

        from functions.etl.split import split

        settings = {'percentage': percentage, 'seed': seed}

        def task_split(df, params):
            return split(df, params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'split',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_split, settings],
             'parent': [self.last_uuid],
             'output': 2,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list,
                   last_uuid=new_state_uuid, settings={'input': 0}), \
            DDF(task_list=self.task_list, last_uuid=new_state_uuid,
                settings={'input': 1})

    def take(self, num):
        """
        Returns the first num rows.

        Is it a Lazy function: No

        :param num: number of rows to retrieve;
        :return: DDF

        :Example:

        >>> ddf1.take(10)
        """

        from functions.etl.sample import take
        settings = {'value': num}

        def task_take(df, params):
            return take(df, params)

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'take',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_take, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             'info': True
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

    def union(self, data2):
        """
        Combine this data set with some other DDF.

        Is it a Lazy function: No

        :param data2:
        :return: DDF

        :Example:

        >>> ddf1.union(ddf2)
        """

        from functions.etl.union import union

        def task_union(df, params):
            return union(df[0], df[1])

        new_state_uuid = self._generate_uuid()
        context.COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'union',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_union, {}],
             'parent': [self.last_uuid,  data2.last_uuid],
             'output': 1,
             'input': 2
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        self._set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self._merge_tasks_list(self.task_list + data2.task_list)
        return DDF(task_list=new_list, last_uuid=new_state_uuid)

