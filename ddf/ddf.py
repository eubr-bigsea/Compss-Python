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
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
import cPickle as pickle

from collections import OrderedDict, deque

import copy

DEBUG = False


class COMPSsContext(object):
    """
    Controls the execution of DDF tasks
    """
    tasks_map = OrderedDict()
    adj_tasks = dict()

    def get_var_by_task(self, variables, uuid):
        """
        Return the variable id which contains the task uuid.

        :param variables:
        :param uuid:
        :return:
        """
        return [i for i, v in enumerate(variables) if uuid in v.task_list]

    def show_workflow(self, tasks_to_in_count):
        """
        Show the final workflow. Only to debug
        :param tasks_to_in_count: list of tasks to be executed in this flow.
        """
        print("Relevant tasks:")
        for uuid in tasks_to_in_count:
            print("\t{} - ({})".format(self.tasks_map[uuid]['name'], uuid[:8]))
        print ("\n")

    def show_tasks(self):
        """
        Show all tasks in the current code. Only to debug.
        :return:
        """
        print("List of all tasks:")
        for k in self.tasks_map:
            print("{} --> {}".format(k[:8], self.tasks_map[k]))
        print ("\n")

    def create_adj_tasks(self, specials=[]):
        """
        Create a Adjacency task list
        Parent --> sons
        :return:
        """

        for t in self.tasks_map:
            parents = self.tasks_map[t]['parent']
            if t in specials:
                for p in parents:
                    if p in specials:
                        if p not in self.adj_tasks:
                            self.adj_tasks[p] = []
                        self.adj_tasks[p].append(t)

        GRAY, BLACK = 0, 1

        def topological(graph):
            order, enter, state = deque(), set(graph), {}

            def dfs(node):
                state[node] = GRAY
                for k in graph.get(node, ()):
                    sk = state.get(k, None)
                    if sk == GRAY:
                        raise ValueError("cycle")
                    if sk == BLACK:
                        continue
                    enter.discard(k)
                    dfs(k)
                order.appendleft(node)
                state[node] = BLACK

            while enter: dfs(enter.pop())
            return order

        for k in self.adj_tasks:
            self.adj_tasks[k] = list(set(self.adj_tasks[k]))
            if DEBUG:
                print "{} ({}) --> {}".format(k, self.tasks_map[k]['name'],
                                              self.adj_tasks[k])

        result = list(topological(self.adj_tasks))

        return result

    def run(self, wanted=-1):
        import gc

        if DEBUG:
            self.show_tasks()

        # mapping all tasks that produce a final result
        action_tasks = []
        if wanted is not -1:
            action_tasks.append(wanted)

        for t in self.tasks_map:
            if self.tasks_map[t]['name'] in ['save', 'sync']:
                action_tasks.append(t)

        if DEBUG:
            print "action:", action_tasks
        # based on that, get the their variables
        variables = []
        n_vars = 0
        for obj in gc.get_objects():
            if isinstance(obj, DDF):
                n_vars += 1
                tasks = obj.task_list
                for k in action_tasks:
                    if k in tasks:
                        variables.append(copy.deepcopy(obj))

        # list all tasks used in these variables
        tasks_to_in_count = list()
        for var in variables:
            tasks_to_in_count.extend(var.task_list)

        # and perform a topological sort to create a DAG
        topological_tasks = self.create_adj_tasks(tasks_to_in_count)
        if DEBUG:
            print "topological_tasks: ", topological_tasks
        # print("Number of variables: {}".format(len(variables)))
        # print("Total of variables: {}".format(n_vars))
        # self.show_workflow(tasks_to_in_count)

        # iterate over all filtered and sorted tasks
        for current_task in topological_tasks:
            # print("\n* Current task: {} ({}) -> {}".format(
            #         self.tasks_map[current_task]['name'], current_task[:8],
            #         self.tasks_map[current_task]['status']))

            if self.tasks_map[current_task]['status'] == 'COMPLETED':
                continue
            # get its variable and related tasks
            if DEBUG:
                print "look for: ", current_task
            id_var = self.get_var_by_task(variables, current_task)
            if len(id_var) == 0:
                raise Exception("\nVariable was deleted")

            id_var = id_var[0]
            tasks_list = variables[id_var].task_list
            id_task = tasks_list.index(current_task)
            tasks_list = tasks_list[id_task:]

            for i_task, child_task in enumerate(tasks_list):
                id_parents = self.tasks_map[child_task]['parent']
                n_input = self.tasks_map[child_task].get('n_input', [-1])

                if self.tasks_map[child_task]['status'] == 'WAIT':
                    if DEBUG:
                        print(" - task {} ({})  parents:{}" \
                              .format(self.tasks_map[child_task]['name'],
                                      child_task[:8], id_parents))

                        print "id_parents: {} and n_input: {}".format(
                                id_parents, n_input)

                    # when has parents: wait all parents tasks be completed
                    if not all([self.tasks_map[p]['status'] == 'COMPLETED'
                                for p in id_parents]):
                        break

                    # get input data from parents
                    inputs = {}
                    for d, (id_p, id_in) in enumerate(zip(id_parents, n_input)):
                        if id_in == -1:
                            id_in = 0
                        inputs[d] = self.tasks_map[id_p]['function'][id_in]
                    variables[id_var].partitions = inputs

                    # end the path
                    if self.tasks_map[child_task]['name'] == 'sync':
                        if DEBUG:
                            print "RUNNING sync ({}) - condition 2." \
                                 .format(child_task)
                        # sync tasks always will have only one parent
                        id_p = id_parents[0]
                        id_in = n_input[0]

                        self.tasks_map[child_task]['function'] = \
                            self.tasks_map[id_p]['function']

                        result = inputs[0]
                        if id_in == -1:
                            self.tasks_map[id_p]['function'][0] = result
                            self.tasks_map[child_task]['function'][0] = result

                        else:
                            self.tasks_map[id_p]['function'][id_in] = result
                            self.tasks_map[child_task]['function'][id_in] = \
                                result

                        self.tasks_map[child_task]['status'] = 'COMPLETED'
                        self.tasks_map[id_p]['status'] = 'COMPLETED'

                        break

                    elif not self.tasks_map[child_task]['lazy']:
                        # Execute f and put result in variables
                        if DEBUG:
                            print "RUNNING {} ({}) - condition 3.".format(
                                self.tasks_map[child_task]['name'], child_task)

                        f = self.tasks_map[child_task]['function']
                        variables[id_var]._task_others(f)

                        self.tasks_map[child_task]['status'] = 'COMPLETED'

                        # output format: {0: ... 1: ....}
                        n_outputs = self.tasks_map[child_task]['output']
                        out = {}
                        if n_outputs == 1:
                            out[0] = variables[id_var].partitions
                        else:
                            for f in range(n_outputs):
                                out[f] = variables[id_var].partitions[f]

                        self.tasks_map[child_task]['function'] = out

                    elif self.tasks_map[child_task]['lazy']:
                        self.run_serial_lazy_tasks(i_task, child_task,
                                                   tasks_list,
                                                   tasks_to_in_count, id_var,
                                                   variables)

    def run_serial_lazy_tasks(self, i_task, child_task, tasks_list,
                              tasks_to_in_count, id_var, variables):
        opt = set()
        opt_functions = []

        if DEBUG:
            print "RUNNING {} ({}) - condition 4.".format(
                self.tasks_map[child_task]['name'], child_task)

        for id_j, task_opt in enumerate(tasks_list[i_task:]):
            if DEBUG:
                print 'Checking lazziness: {} -> {}'.format(
                    task_opt[:8],
                    self.tasks_map[task_opt]['name'])

            opt.add(task_opt)
            opt_functions.append(
                    self.tasks_map[task_opt]['function'])

            if (i_task + id_j + 1) < len(tasks_list):
                next_task = tasks_list[i_task + id_j + 1]

                if tasks_to_in_count.count(task_opt) != \
                        tasks_to_in_count.count(next_task):
                    break

                if not all([self.tasks_map[task_opt]['lazy'],
                            self.tasks_map[next_task]['lazy']]):
                    break

        if DEBUG:
            print "Stages (optimized): {}".format(opt)
            print "opt_functions", opt_functions
        variables[id_var]._perform(opt_functions)

        out = {}
        for o in opt:
            self.tasks_map[o]['status'] = 'COMPLETED'

            # output format: {0: ... 1: ....}
            n_outputs = self.tasks_map[o]['output']

            if n_outputs == 1:
                out[0] = variables[id_var].partitions
            else:
                for f in range(n_outputs):
                    out[f] = variables[id_var].partitions[f]

            self.tasks_map[o]['function'] = out

            if DEBUG:
                print "{} ({}) is COMPLETED - condition 4." \
                    .format(self.tasks_map[o]['name'], o[:8])

        variables[id_var].partitions = out


@task(returns=1)
def task_bundle(data, functions, id_frag):

    for f in functions:
        function, settings = f
        # Used only in save
        if isinstance(settings, dict):
            settings['id_frag'] = id_frag
        data = function(data, settings)

    return data


class DDF(object):
    """
    Distributed DataFrame Handler.

    Should distribute the data and run tasks for each partition.
    """

    def __init__(self, **kwargs):
        super(DDF, self).__init__()

        task_list = kwargs.get('task_list', None)
        last_uuid = kwargs.get('last_uuid', 'init')
        self.settings = kwargs.get('settings', {'input': -1})

        self.schema = list()
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

            COMPSsContext.tasks_map[last_uuid] = \
                {'name': 'init',
                 'lazy': False,
                 'input': 0,
                 'parent': [],
                 'status': 'COMPLETED'
                 }

        self.last_uuid = last_uuid

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
        while new_state_uuid in COMPSsContext.tasks_map:
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

        if 'n_input' not in COMPSsContext.tasks_map[state_uuid]:
            COMPSsContext.tasks_map[state_uuid]['n_input'] = []
        COMPSsContext.tasks_map[state_uuid]['n_input'].append(idx)

    def _task_others(self, f):
        """
        Used to execute all non-lazy functions.

        :param f: a list that contains the current task and its parameters.
        :return:
        """

        function, settings = f
        if len(self.partitions) > 1:
            self.partitions = [self.partitions[k] for k in self.partitions]
        else:
            self.partitions = self.partitions[0]

        self.partitions = function(self.partitions, settings)

    def _perform(self, opt):

        """
        Used to execute a group of lazy tasks. This method submit
        multiple 'task_bundle', one for each data fragment.

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

        future_objects = []
        for idfrag, p in enumerate(tmp):
            future_objects.append(task_bundle(p, opt, idfrag))

        self.partitions = future_objects

    def load_text(self, filename, num_of_parts=4, header=True, sep=',',
                  storage='hdfs'):
        """
        Create a DDF from a commom file system or from HDFS.

        :param filename: Input file name;
        :param num_of_parts: number of partitions (default, 4);
        :param header: Use the first line as DataFrame header (default, True);
        :param sep: separator delimiter (default, ',');
        :param storage: *'hdfs'* to use HDFS as storage or *'fs'* to use the
         common file sytem;
        :return: DDF.

        :Example:

        >>> ddf1 = DDF().load_text('/titanic.csv', num_of_parts=4)
        """
        if storage not in ['hdfs', 'fs']:
            raise Exception('`hdfs` and `fs` storage are supported.')

        if storage == 'hdfs':
            from functions.etl.read_data import ReadOperationHDFS

            settings = dict()
            settings['port'] = 9000
            settings['host'] = 'localhost'
            settings['separator'] = ','
            settings['header'] = header
            settings['separator'] = sep

            self.partitions, info = ReadOperationHDFS() \
                .transform(filename, settings, num_of_parts)

            new_state_uuid = self._generate_uuid()
            COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'load_hdfs',
                 'status': 'COMPLETED',
                 'lazy': False,
                 'function': {0: self.partitions},
                 'output': 1,
                 'input': 0,
                 'parent': [self.last_uuid]
                 }

            self.partial_sizes.append(info)
            self._set_n_input(new_state_uuid, self.settings['input'])
            return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

        else:

            from functions.etl.read_data import ReadOperationFs

            settings = dict()
            settings['port'] = 9000
            settings['host'] = 'localhost'
            settings['separator'] = ','
            settings['header'] = header
            settings['separator'] = sep

            self.partitions, info = ReadOperationFs()\
                .transform(filename, settings, num_of_parts)

            new_state_uuid = self._generate_uuid()
            COMPSsContext.tasks_map[new_state_uuid] = \
                {'name': 'load_fs',
                 'status': 'COMPLETED',
                 'lazy': False,
                 'function': {0: self.partitions},
                 'output': 1,
                 'input': 0,
                 'parent': [self.last_uuid]
                 }

            self.partial_sizes.append(info)
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

        from functions.etl.data_functions import Partitionize

        self.partitions = Partitionize(df, num_of_parts)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'parallelize',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: self.partitions},
             'output': 1, 'input': 0,
             'parent': [self.last_uuid]
             }

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

        from functions.geo import ReadShapeFileOperation

        self.partitions = \
            ReadShapeFileOperation().transform(settings, num_of_parts)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'load_shapefile',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: self.partitions},
             'output': 1, 'input': 0,
             'parent': [self.last_uuid]
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

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
             'lazy': False,
             'parent': [self.last_uuid],
             'function': [[]]
             }

        self._set_n_input(new_state_uuid, self.settings['input'])

        tmp = DDF(task_list=self.task_list, last_uuid=new_state_uuid,
                  settings=self.settings)._run_compss_context(new_state_uuid)

        return tmp

    def _run_compss_context(self, wanted=None):
        COMPSsContext().run(wanted)
        return self

    # TODO: CHECK it
    def num_of_partitions(self):
        """
        Returns the number of data partitions (Task parallelism).

        :return: integer

        :Example:

        >>> print ddf1.num_of_partitions()
        """
        size = len(COMPSsContext().tasks_map[self.last_uuid]['function'])
        return size

    def _schema(self):

        return self.schema

    def count(self):
        """
        Return a number of rows in this DDF.

        :return: integer

        :Example:

        >>> print ddf1.count()
        """

        def task_count(df, params):
            return len(df)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'count',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_count, {}],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }
        self._set_n_input(new_state_uuid, self.settings['input'])

        tmp = DDF(task_list=self.task_list, last_uuid=new_state_uuid).cache()

        result = COMPSsContext.tasks_map[new_state_uuid]['function']
        res = compss_wait_on(result[0])
        del tmp
        res = sum(res)

        return res

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

        from functions.etl.attributes_changer \
            import AttributesChangerOperation

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

        def task_with_column(df, params):
            return AttributesChangerOperation().transform_serial(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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
        COMPSsContext.tasks_map[new_state_uuid] = \
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
        COMPSsContext.tasks_map[new_state_uuid] = \
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
        COMPSsContext.tasks_map[new_state_uuid] = \
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

    # def cross_join(self, data2):
    #     """
    #     Returns the cartesian product with another DDF.
    #
    #     :param data2: Right side of the cartesian product;
    #     :return: DDF.
    #
    #     :Example:
    #
    #     >>> ddf1.cross_join(ddf2)
    #     """
    #     from functions.etl.cross_join import CrossJoinOperation
    #
    #     def task_cross_join(df, params):
    #         return CrossJoinOperation().transform(df[0], df[1])
    #
    #     new_state_uuid = self._generate_uuid()
    #     COMPSsContext.tasks_map[new_state_uuid] = \
    #         {'name': 'cross_join',
    #          'status': 'WAIT',
    #          'lazy': False,
    #          'function': [task_cross_join, {}],
    #          'parent': [self.last_uuid, data2.last_uuid],
    #          'output': 1,
    #          'input': 2
    #          }
    #
    #     self._set_n_input(new_state_uuid, self.settings['input'])
    #     self._set_n_input(new_state_uuid, data2.settings['input'])
    #     new_list = self._merge_tasks_list(self.task_list + data2.task_list)
    #     return DDF(task_list=new_list, last_uuid=new_state_uuid)

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
        from functions.etl.difference import DifferenceOperation

        def task_difference(df, params):
            return DifferenceOperation()\
                .transform(df[0], df[1], len(df[0]))

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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
        from functions.etl.distinct import DistinctOperation

        def task_distinct(df, params):
            return DistinctOperation().transform(df, params, len(df))

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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

        def task_drop(df, cols):
            return df.drop(cols, axis=1)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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

        def task_filter(df, query):
            return df.query(query)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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
        COMPSsContext.tasks_map[new_state_uuid] = \
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
        COMPSsContext.tasks_map[new_state_uuid] = \
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
        COMPSsContext.tasks_map[new_state_uuid] = \
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

        from functions.etl.replace_values import ReplaceValuesOperation

        settings = dict()
        settings['replaces'] = replaces
        settings['subset'] = subset
        settings = ReplaceValuesOperation().preprocessing(settings)

        def task_replace(df, params):
            return ReplaceValuesOperation().transform_serial(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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

        from functions.etl.sample import SampleOperation
        settings = dict()
        settings['seed'] = seed

        if value:
            """Sample a N random records"""
            settings['type'] = 'value'
            if isinstance(value, float):
                settings['per_value'] = value
            else:
                settings['int_value'] = value

        else:
            """Sample a random amount of records"""
            settings['type'] = 'percent'

        def task_sample(df, params):
            return SampleOperation().transform(df, params, len(df))

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sample',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_sample, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
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
        COMPSsContext.tasks_map[new_state_uuid] = \
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

    def select(self, columns):
        """
        Projects a set of expressions and returns a new DDF.

        Is it a Lazy function: Yes

        :param columns: list of column names (string);
        :return: DDF

        :Example:

        >>> ddf1.select(['col_1', 'col_2'])
        """

        def task_select(df, fields):
            # remove the columns that not in list1
            fields = [field for field in fields if field in df.columns]
            if len(fields) == 0:
                raise Exception("The columns passed as parameters "
                                "do not belong to this DataFrame.")
            return df[fields]

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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
        cached = False

        for _ in range(2):
            if COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
                n_input = COMPSsContext.tasks_map[self.last_uuid]['n_input'][0]
                if n_input == -1:
                    n_input = 0
                self.partitions = \
                    COMPSsContext.tasks_map[self.last_uuid]['function'][n_input]
                cached = True
            else:
                self.cache()

        if not cached:
            print "last_uuid: {} ".format(self.last_uuid)
            print self.task_list
            print COMPSsContext.tasks_map[self.last_uuid]
            raise Exception("ERROR - toPandas - not cached:")

        res = compss_wait_on(self.partitions)
        n_input = self.settings['input']
        if n_input == -1:
            n_input = 0

        COMPSsContext.tasks_map[self.last_uuid]['function'][n_input] = res
        COMPSsContext.tasks_map[last_last_uuid]['function'][n_input] = res

        df = pd.concat(res, sort=True)[:abs(n)]
        df.reset_index(drop=True, inplace=True)
        return df

    def sort(self, cols,  ascending=[]):
        """
        Returns a sorted DDF by the specified column(s).

        Is it a Lazy function: No

        :param cols: list of columns to be sorted;
        :param ascending: list indicating whether the sort order
            is ascending (True) for each column;
        :return: DDF

        :Example:

        >>> dd1.sort(['col_1', 'col_2'], ascending=['True', 'False'])
        """

        from functions.etl.sort import SortOperation

        settings = dict()
        settings['columns'] = cols
        settings['ascending'] = ascending

        def task_sort(df, params):
            return SortOperation().transform(df, params, len(df))

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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

        :param percentage:  percentage to split the data (default, 0.5);
        :param seed: optional, seed in case of deterministic random operation;
        :return: DDF

        :Example:

        >>> ddf2a, ddf2b = ddf1.split(0.5)
        """

        from functions.etl.split import SplitOperation

        settings = dict()
        settings['percentage'] = percentage
        settings['seed'] = seed

        def task_split(df, params):
            return SplitOperation().transform(df, params, len(df))

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'split',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_split, settings],
             'parent': [self.last_uuid],
             'output': 2, 'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list,
                   last_uuid=new_state_uuid, settings={'input': 0}), \
               DDF(task_list=self.task_list,
                   last_uuid=new_state_uuid, settings={'input': 1})

    def take(self, num):
        """
        Returns the first num rows.

        Is it a Lazy function: No

        :param num: number of rows to retrieve;
        :return: DDF

        :Example:

        >>> ddf1.take(10)
        """

        from functions.etl.sample import SampleOperation
        settings = dict()
        settings['type'] = 'head'
        settings['int_value'] = num

        def task_take(df, params):
            return SampleOperation().transform(df, params, len(df))

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'take',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_take, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self._set_n_input(new_state_uuid, self.settings['input'])
        return DDF(task_list=self.task_list, last_uuid=new_state_uuid)

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

        from functions.etl.transform import TransformOperation

        def task_map(df, params):
            return TransformOperation().transform(df, params)

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'map',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_map, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1}

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

        from functions.etl.union import UnionOperation

        def task_union(df, params):
            return UnionOperation().transform(df[0], df[1])

        new_state_uuid = self._generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
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


class ModelDDF(object):

    def __init__(self):

        self.settings = dict()
        self.model = []
        self.name = ''

    def save_model(self, filepath, storage='hdfs', overwrite=True,
                   namenode='localhost', port=9000):
        """
        Save a machine learning model as a binary file in a storage.

        :param filepath: The output absolute path name;
        :param storage: *'hdfs'* to save in HDFS storage or *'fs'* to save in
         common file system;
        :param overwrite: Overwrite if file already exists (default, True);
        :param namenode: IP or DNS address to NameNode (default, *'localhost'*);
        :param port: NameNode port (default, 9000);
        :return: self

        :Example:

        >>> ml_model.save_model('/trained_model')
        """
        if storage not in ['hdfs', 'fs']:
            raise Exception('Only `hdfs` and `fs` storage are supported.')

        if storage == 'hdfs':
            save_model_hdfs(self.model, filepath, namenode,
                            port, overwrite)
        else:
            save_model_fs(self.model, self.settings)

        return self

    def load_model(self, filepath, storage='hdfs', namenode='localhost',
                   port=9000):
        """
        Load a machine learning model from a binary file in a storage.

        :param filepath: The absolute path name;
        :param storage: *'hdfs'* to load from HDFS storage or *'fs'* to load
         from common file system;
        :param storage: *'hdfs'* to save in HDFS storage or *'fs'* to save in
         common file system;
        :param namenode: IP or DNS address to NameNode (default, *'localhost'*).
         Note: Only if storage is *'hdfs'*;
        :param port: NameNode port (default, 9000). Note: Only if storage is
         *'hdfs'*;
        :return: self

        :Example:

        >>> ml_model = ml_algorithm().load_model('/saved_model')
        """
        if storage not in ['hdfs', 'fs']:
            raise Exception('Only `hdfs` and `fs` storage are supported.')

        if storage == 'hdfs':
            self.model = load_model_hdfs(filepath, namenode, port)
        else:
            self.model = load_model_fs(filepath)

        return self


#@task(returns=1)
def save_model_hdfs(model, path, namenode='localhost', port=9000,
                    overwrite=True):
    """
    Save a machine learning model as a binary file in a HDFS storage.

    :param model: Model to be storaged in HDFS;
    :param path: The path of the file from the '/' of the HDFS;
    :param namenode: The host of the Namenode HDFS; (default, 'localhost')
    :param port: NameNode port (default, 9000).
    :param overwrite: Overwrite if file already exists (default, True);
    """
    from hdfspycompss.HDFS import HDFS
    dfs = HDFS(host=namenode, port=port)

    if dfs.exist(path) and not overwrite:
        raise Exception("File already exists in this source.")

    to_save = pickle.dumps(model, 0)
    dfs.writeBlock(path, to_save, append=False, overwrite=True)
    return [-1]


#@task(filename=FILE_OUT)
def save_model_fs(model, filepath):
    """
    Save a machine learning model as a binary file in a common file system.

    Save a machine learning model into a file.
    :param filepath: Absolute file path;
    :param model: The model to be saved
    """
    with open(filepath, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


#@task(returns=1)
def load_model_hdfs(filepath, namenode='localhost', port=9000):
    """
    Load a machine learning model from a HDFS source.

    :param filepath: The path of the file from the '/' of the HDFS;
    :param namenode: The host of the Namenode HDFS; (default, 'localhost')
    :param port: NameNode port (default, 9000).
    :return: Returns a model
    """
    from hdfspycompss.HDFS import HDFS
    from hdfspycompss.Block import Block

    dfs = HDFS(host=namenode, port=port)
    blk = dfs.findNBlocks(filepath, 1)
    to_load = Block(blk).readBinary()
    model = None
    if len(to_load) > 0:
        model = pickle.loads(to_load)
    return model


#@task(returns=1, filename=FILE_IN)
def load_model_fs(filepath):
    """
    Load a machine learning model from a common file system.

    :param filepath: Absolute file path;
    :return: Returns a model.
    """
    with open(filepath, 'rb') as input:
        model = pickle.load(input)
    return model
