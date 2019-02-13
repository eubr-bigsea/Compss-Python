#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import uuid

import pandas as pd
import numpy as np
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on

from pycompss.api.parameter import FILE_IN, FILE_OUT
import cPickle as pickle

from collections import OrderedDict, deque

import copy


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
        :param list_uuid:
        :return:
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

        print "adj_tasks:"
        for k in self.adj_tasks:
            self.adj_tasks[k] = list(set(self.adj_tasks[k]))
            print "{} ({}) --> {}".format(k, self.tasks_map[k]['name'], self.adj_tasks[k])

        result = list(topological(self.adj_tasks))

        return result

    def run(self, wanted=-1):
        import gc

        self.show_tasks()

        # mapping all tasks that produce a final result
        action_tasks = []
        if wanted is not -1:
            action_tasks.append(wanted)

        for t in self.tasks_map:
            if self.tasks_map[t]['name'] in ['save', 'sync']:
                action_tasks.append(t)

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
            # print "- var:", var.task_list
            tasks_to_in_count.extend(var.task_list)

        # and perform a topological sort to create a DAG
        topological_tasks = self.create_adj_tasks(tasks_to_in_count)

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
            print "look for: ", current_task
            id_var = self.get_var_by_task(variables, current_task)
            if len(id_var) == 0:
                raise Exception("\nvariable was deleted")
                break

            id_var = id_var[0]
            tasks_list = variables[id_var].task_list
            id_task = tasks_list.index(current_task)
            # print ("* var:{} task:{} em {} ".format(id_var, id_task, tasks_list))

            tasks_list = tasks_list[id_task:]
            for i_task, child_task in enumerate(tasks_list):
                id_parents = self.tasks_map[child_task]['parent']
                n_input = self.tasks_map[child_task].get('n_input', [-1])

                if self.tasks_map[child_task]['status'] == 'WAIT':
                    print(" - task {} ({})  parents:{}" \
                          .format(self.tasks_map[child_task]['name'],
                                  child_task[:8], id_parents))

                    print "id_parents: {} and n_input: {}".format(
                            id_parents, n_input)

                    # when has parents: wait all parents tasks be completed
                    if not all([self.tasks_map[p]['status'] == 'COMPLETED'
                                for p in id_parents]):
                        print "WAITING FOR A PARENT BE COMPLETED"
                        break

                    # get input data from parents
                    inputs = {}
                    for d, (id_p, id_in) in enumerate(zip(id_parents, n_input)):
                        print "B:", self.tasks_map[id_p]['function']
                        if id_in == -1:
                            id_in = 0
                        inputs[d] = self.tasks_map[id_p]['function'][id_in]
                    variables[id_var].partitions = inputs
                    print "----\n New INPUTS: {}\n".format(inputs)

                    # end the path
                    if self.tasks_map[child_task]['name'] == 'sync':
                        print "RUNNING sync ({}) - condition 2." \
                                 .format(child_task)
                        # sync tasks always will have only one parent
                        id_p = id_parents[0]
                        id_in = n_input[0]
                        # print "parent", self.tasks_map[id_p]

                        self.tasks_map[child_task]['function'] = \
                            self.tasks_map[id_p]['function']

                        result = inputs[0]
                        if id_in == -1:
                            print "HERE1"
                            self.tasks_map[id_p]['function'][0] = result

                            self.tasks_map[child_task]['function'][0] = result
                                # variables[id_var].partitions[0]
                        else:
                            print "HERE2"
                            self.tasks_map[id_p]['function'][id_in] = result
                            self.tasks_map[child_task]['function'][id_in] = \
                                result
                                #variables[id_var].partitions[0]

                        # self.tasks_map[child_task]['function'] = inputs

                        self.tasks_map[child_task]['status'] = 'COMPLETED'
                        self.tasks_map[id_p]['status'] = 'COMPLETED'

                        #print "current task", self.tasks_map[child_task]
                        #print "current parent", self.tasks_map[id_p]

                        break

                    elif not self.tasks_map[child_task]['lazy']:
                        # Execute f and put result in variables

                        print "RUNNING {} ({}) - condition 3.".format(
                                self.tasks_map[child_task]['name'], child_task)

                        f = self.tasks_map[child_task]['function']
                        variables[id_var].task_others(f)

                        self.tasks_map[child_task]['status'] = 'COMPLETED'

                        # o resultado da task_map no formato {0: ... 1: ....}
                        # resultado salvo unicamente na propria key
                        n_outputs = self.tasks_map[child_task]['output']
                        out = {}
                        if n_outputs == 1:
                            print 'single output'
                            out[0] = variables[id_var].partitions
                        else:
                            print 'multiple outputs'
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

        print "RUNNING {} ({}) - condition 4.".format(
                self.tasks_map[child_task]['name'], child_task)

        for id_j, task_opt in enumerate(tasks_list[i_task:]):
            print 'Checking lazziness: {} -> {}'.format(
                    task_opt[:8],
                    self.tasks_map[task_opt]['name'])

            opt.add(task_opt)
            opt_functions.append(
                    self.tasks_map[task_opt]['function'])

            if (i_task + id_j + 1) < len(tasks_list):
                next_task = tasks_list[i_task + id_j + 1]

                print "{} vs {}" \
                    .format(self.tasks_map[task_opt]['name'],
                            self.tasks_map[next_task]['name'])

                if tasks_to_in_count.count(task_opt) != \
                        tasks_to_in_count.count(next_task):
                    print "opt - exit 1"
                    break

                if not all([self.tasks_map[task_opt]['lazy'],
                            self.tasks_map[next_task]['lazy']]):
                    print "opt - exit 2"
                    break

        print "Stages (optimized): {}".format(opt)
        print "opt_functions", opt_functions
        variables[id_var].perform(opt_functions)

        out = {}
        for o in opt:
            self.tasks_map[o]['status'] = 'COMPLETED'

            # o resultado da task_map no formato {0: ... 1: ....}
            n_outputs = self.tasks_map[o]['output']

            if n_outputs == 1:
                out[0] = variables[id_var].partitions
            else:
                for f in range(n_outputs):
                    out[f] = variables[id_var].partitions[f]

            self.tasks_map[o]['function'] = out

            # self.tasks_map[o]['function'] = \
            #     variables[id_var].partitions

            print "{} ({}) is COMPLETED - condition 4." \
                .format(self.tasks_map[o]['name'], o[:8])

            print self.tasks_map[o]
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

    def __init__(self, task_list=None,
                 last_uuid='init', settings={'input': -1}):
        super(DDF, self).__init__()

        self.schema = list()
        self.opt = OrderedDict()
        self.partial_sizes = list()
        self.settings = settings
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
    def merge_tasks_list(seq):
        """
        Merge two list of tasks removing duplicated tasks

        :param seq: list with possible duplicated elements
        :return:
        """
        seen = set()
        return [x for x in seq if x not in seen and not seen.add(x)]

    @staticmethod
    def generate_uuid():
        """
        Generate a unique id
        :return: uuid
        """
        new_state_uuid = str(uuid.uuid4())
        while new_state_uuid in COMPSsContext.tasks_map:
            new_state_uuid = str(uuid.uuid4())
        return new_state_uuid

    @staticmethod
    def set_n_input(state_uuid, idx):
        """
        Method to inform the index of the input data

        :param state_uuid: id of the current task
        :param idx: idx of input data
        :return:
        """

        if 'n_input' not in COMPSsContext.tasks_map[state_uuid]:
            COMPSsContext.tasks_map[state_uuid]['n_input'] = []
        COMPSsContext.tasks_map[state_uuid]['n_input'].append(idx)

    def task_others(self, f):
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

    def perform(self, opt):
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

    def load_fs(self, filename, num_of_parts=4, header=True, sep=','):

        from functions.etl.read_data import ReadOperationHDFS

        settings = dict()
        settings['port'] = 9000
        settings['host'] = 'localhost'
        settings['separator'] = ','
        settings['header'] = header
        settings['separator'] = sep

        self.partitions, info = ReadOperationHDFS()\
            .transform(filename, settings, num_of_parts)

        new_state_uuid = self.generate_uuid()
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
        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def parallelize(self, df, num_of_parts=4):
        """
        Use the iterator and create the partitions of this DDS.
        :param df:
        :param num_of_parts:
        :return:

        """

        from functions.etl.data_functions import Partitionize

        self.partitions = Partitionize(df, num_of_parts)

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'parallelize',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: self.partitions},
             'output': 1, 'input': 0,
             'parent': [self.last_uuid]
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def load_shapefile(self, shp_path, dbf_path, polygon='points',
                       attributes=[], num_of_parts=4):
        """

        :param shp_path: Path to the shapefile (.shp)
        :param dbf_path: Path to the shapefile (.dbf)
        :param polygon: Alias to the new column to store the
                polygon coordenates (default, 'points');
        :param attributes: List of attributes to keep in the dataframe,
                empty to use all fields;
        :param num_of_parts: The number of fragments;
        :return:

        Note: pip install pyshp

        """

        settings = dict()
        settings['shp_path'] = shp_path
        settings['dbf_path'] = dbf_path
        settings['polygon'] = polygon
        settings['attributes'] = attributes

        from functions.geo import ReadShapeFileOperation

        self.partitions = \
            ReadShapeFileOperation().transform(settings, num_of_parts)

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'load_shapefile',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: self.partitions},
             'output': 1, 'input': 0,
             'parent': [self.last_uuid]
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def collect(self):
        if COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
            self.partitions = \
                COMPSsContext.tasks_map[self.last_uuid]['function'][0]

        self.partitions = compss_wait_on(self.partitions)

    def cache(self):
        """
        Compute all tasks until the current state

        :return:

        """

        # TODO: no momento, é necessario para lidar com split
        # if COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
        #     print 'cache skipped'
        #     return self

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sync',
             'status': 'WAIT',
             'lazy': False,
             'parent': [self.last_uuid],
             'function': [[]]
             }

        self.set_n_input(new_state_uuid, self.settings['input'])

        print "doit", new_state_uuid
        tmp = DDF(self.task_list, new_state_uuid,
                  self.settings).doit(new_state_uuid)

        return tmp

    def doit(self, wanted=None):
        COMPSsContext().run(wanted)
        return self

    # TODO: CHECK it
    def num_of_partitions(self):
        """
        Get the total amount of partitions
        :return: int

        """
        size = len(COMPSsContext().tasks_map[self.last_uuid]['function'])
        return size

    def schema(self):

        return self.schema

    def show(self, n=20):
        """
        Returns the DDF contents in a concatenated pandas's DataFrame.

        :return:
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
        return pd.concat(res)[:abs(n)]

    def count(self):
        """
        :return: total number of elements
        """

        def task_count(df, params):
            return len(df)

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'count',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_count, {}],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }
        self.set_n_input(new_state_uuid, self.settings['input'])

        tmp = DDF(self.task_list, new_state_uuid).cache()

        result = COMPSsContext.tasks_map[new_state_uuid]['function']
        res = compss_wait_on(result[0])
        del tmp
        res = sum(res)

        return res

    def with_column(self, old_column, new_column=None, cast=None):
        """
        Rename or change the data's type of some columns.

        Lazy function

        :param old_column:
        :param new_column:
        :param cast:
        :return:
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

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'with_column',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_with_column, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])

        return DDF(self.task_list, new_state_uuid)

    # TODO: CHECK it
    def add_column(self, data, suffixes=["_l","_r"]):
        """

        :param data:
        :param suffixes:
        :return:
        """

        # se for size flexivel, entao isso vai ser complicado
        return []

    def aggregation(self, group_by, exprs, aliases):
        """

        :param group_by:
        :param exprs:
        :return:
        """

        settings = dict()
        settings['columns'] = group_by
        # settings['exprs'] = exprs  exprs = {'date': {'COUNT': 'count'}}
        settings['operation'] = exprs
        settings['aliases'] = aliases

        from functions.etl.aggregation import AggregationOperation

        def task_aggregation(df, params):
            return AggregationOperation().transform(df, params, len(df))

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'aggregation',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_aggregation, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def clean_missing(self, columns, mode='REMOVE_ROW', value=None):

        from functions.etl.clean_missing import CleanMissingOperation

        settings = dict()
        settings['attributes'] = columns
        settings['cleaning_mode'] = mode
        settings['value'] = value

        def task_clean_missing(df, params):
            return CleanMissingOperation().transform(df, params, len(df))

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'clean_missing',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_clean_missing, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def difference(self, data2):
        """
        Returns a new set with containing rows in the first frame but not
        in the second one.

        :param data2: second DDF
        :return:
        """
        from functions.etl.difference import DifferenceOperation

        def task_difference(df, params):
            return DifferenceOperation()\
                .transform(df[0], df[1], len(df[0]))

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'difference',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_difference, {}],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        self.set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self.merge_tasks_list(self.task_list + data2.task_list)
        return DDF(new_list, new_state_uuid)

    def distinct(self, cols):
        """
        Returns a new DDF with non duplicated rows

        :param cols: subset of columns to consider
        :return:
        """
        from functions.etl.distinct import DistinctOperation

        def task_distinct(df, params):
            return DistinctOperation().transform(df, params, len(df))

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'distinct',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_distinct, cols],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def drop(self, columns):
        """
        Perform a partial drop operation.
        Lazy function

        :param columns:
        :return:
        """

        def task_drop(df, cols):
            return df.drop(cols, axis=1)

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'drop',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_drop, columns],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def filter(self, expr):
        """
        Filter elements of this data set.

        Lazy function
        :param expr: A filtering function
        :return:

        """

        def task_filter(df, query):
            return df.query(query)

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'filter',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_filter, expr],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def geo_within(self, shp_object, lat_col, lon_col, polygon,
                   attributes=[], alias='_shp'):
        """

        :param shp_object: The dataframe created by the function ReadShapeFile;
        :param lat_col: Column which represents the Latitute field in the data;
        :param lon_col: Column which represents the Longitude field in the data;
        :param polygon: Field in shp_object where is store the
            coordinates of each sector;
        :param attributes: Attributes to retrieve from shapefile,
            empty to all (default, empty);
        :param alias: Alias for shapefile attributes
            (default, 'sector_position');

        """

        from functions.geo import GeoWithinOperation

        settings = dict()
        settings['lat_col'] = lat_col
        settings['lon_col'] = lon_col
        settings['attributes'] = attributes
        settings['polygon'] = polygon
        settings['alias'] = alias

        def task_geo_within(df, params):
            return GeoWithinOperation().transform(df[0], df[1], params)

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'geo_within',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_geo_within, settings],
             'parent': [self.last_uuid, shp_object.last_uuid],
             'output': 1,
             'input': 2
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        self.set_n_input(new_state_uuid, shp_object.settings['input'])
        new_list = self.merge_tasks_list(self.task_list + shp_object.task_list)
        return DDF(new_list, new_state_uuid)

    def intersect(self, data2):
        """

        :param data2:
        :return:
        """

        from functions.etl.intersect import IntersectionOperation

        def task_intersect(df, params):
            return IntersectionOperation()\
                .transform(df[0], df[1], len(df[0]))

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'intersect',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_intersect, {}],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1,
             'input': 2
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        self.set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self.merge_tasks_list(self.task_list + data2.task_list)
        return DDF(new_list, new_state_uuid)

    def join(self, data2, key1=[], key2=[], mode='inner',
             suffixes=['_l', '_r'], keep_keys=False,
             case=True, sort=True):

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

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'join',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_join, settings],
             'parent': [self.last_uuid, data2.last_uuid],
             'output': 1, 'input': 2}

        self.set_n_input(new_state_uuid, self.settings['input'])
        self.set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self.merge_tasks_list(self.task_list + data2.task_list)
        return DDF(new_list, new_state_uuid)

    def replace(self, replaces, subset=None):
        """

        Lazy function

        :param replaces:
        :param subset:
        :return:
        """

        from functions.etl.replace_values import ReplaceValuesOperation

        settings = dict()
        settings['replaces'] = replaces
        settings = ReplaceValuesOperation().preprocessing(settings)

        def task_replace(df, params):
            return ReplaceValuesOperation().transform_serial(df, params)

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'replace',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_replace, settings],
             'parent': [self.last_uuid],
             'output': 1, 'input': 1}

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def sample(self, value=None, seed=None):
        """
        Returns a sampled subset.

        :param value: None to sample a random amount of records (default),
            a integer or float N to sample a N random records;
        :param seed: pptional, seed for the random operation.
        :return
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

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sample',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_sample, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1,
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def save(self, filename, format='csv', storage='hdfs',
             header=True, mode='overwrite'):
        """
        Save the data in the storage.

        Lazy function.

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

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'save',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_save, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def select(self, columns):
        """
        Perform a projection of selected columns.

        Lazy function
        :param columns: list of columns to be selected
        :return:
        """

        def task_select(df, fields):
            # remove the columns that not in list1
            fields = [field for field in fields if field in df.columns]
            if len(fields) == 0:
                raise Exception("The columns passed as parameters "
                                "do not belong to this DataFrame.")
            return df[fields]

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'select',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_select, columns],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def sort(self, cols,  ascending=[]):
        """
        Returns a sorted DDF by the specified column(s).

        :param cols: list of columns to be sorted;
        :param ascending: list indicating whether the sort order
            is ascending (True) for each column;
        :return:
        """

        from functions.etl.sort import SortOperation

        settings = dict()
        settings['columns'] = cols
        settings['ascending'] = ascending

        def task_sort(df, params):
            return SortOperation().transform(df, params, len(df))

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'sort',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_sort, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def split(self, percentage=0.5, seed=None):
        """
        Randomly splits a DDF into two DDF

        :param percentage:  percentage to split the data (default, 0.5);
        :param seed: optional, seed in case of deterministic random operation;
        :return:
        """

        from functions.etl.split import SplitOperation

        settings = dict()
        settings['percentage'] = percentage
        settings['seed'] = seed

        def task_split(df, params):
            return SplitOperation().transform(df, params, len(df))

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'split',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_split, settings],
             'parent': [self.last_uuid],
             'output': 2, 'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return \
            DDF(self.task_list, new_state_uuid, {'input': 0}),\
            DDF(self.task_list, new_state_uuid, {'input': 1})

    def take(self, num):
        """
        Returns the first num rows.

        :param num: number of rows to retrieve;
        :return:
        """

        from functions.etl.sample import SampleOperation
        settings = dict()
        settings['type'] = 'head'
        settings['int_value'] = num

        def task_take(df, params):
            return SampleOperation().transform(df, params, len(df))

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'take',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_take, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def transform(self, f, alias):
        """
        Apply a function to each row of this data set.

        Lazy function.

        :param f: function that will take each element of this data set as a
                  parameter
        :param alias: name of column to put the result
        :return:
        """

        settings = {'function': f, 'alias': alias}

        from functions.etl.transform import TransformOperation

        def task_transform(df, params):
            return TransformOperation().transform(df, params)

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'transform',
             'status': 'WAIT',
             'lazy': True,
             'function': [task_transform, settings],
             'parent': [self.last_uuid],
             'output': 1,
             'input': 1}

        self.set_n_input(new_state_uuid, self.settings['input'])
        return DDF(self.task_list, new_state_uuid)

    def union(self, data2):
        """
        Combine this data set with some other DDS data.

        Nao eh tao simples assim, tem q sincronizar com as colunas

        :param data2:
        :return:
        """

        from functions.etl.union import UnionOperation

        def task_union(df, params):
            return UnionOperation().transform(df[0], df[1])

        new_state_uuid = self.generate_uuid()
        COMPSsContext.tasks_map[new_state_uuid] = \
            {'name': 'union',
             'status': 'WAIT',
             'lazy': False,
             'function': [task_union, {}],
             'parent': [self.last_uuid,  data2.last_uuid],
             'output': 1,
             'input': 2
             }

        self.set_n_input(new_state_uuid, self.settings['input'])
        self.set_n_input(new_state_uuid, data2.settings['input'])
        new_list = self.merge_tasks_list(self.task_list + data2.task_list)
        return DDF(new_list, new_state_uuid)


class ModelDDS(object):

    def __init__(self):

        self.settings = dict()
        self.model = []
        self.name = ''

    def save(self, filename, storage='hdfs'):
        if storage == 'hdfs':
            return save_model_hdfs(self.model[0], self.settings)
        else:
            return save_model_fs(self.model[0], self.settings)

    def load(self, filename, storage='hdfs'):

        if storage == 'hdfs':
            self.model = [load_model_hdfs(self.settings)]
        else:
            self.model = [load_model_fs(filename)]

        return self


@task(returns=list)
def save_model_hdfs(model, settings):
    """SaveModelToHDFS.
    :param settings:  A dictionary with:
        - path:       The path of the file from the '/' of the HDFS;
        - host:       The host of the Namenode HDFS; (default, 'default')
        - port:       Port of the Namenode HDFS; (default, 0)
        - overwrite:  True if overwrite in case of colision name,
                      False to raise a error.
    """
    from hdfspycompss.HDFS import HDFS
    host = settings.get('host', 'localhost')
    port = settings.get('port', 9000)
    dfs = HDFS(host=host, port=port)

    overwrite = settings.get('overwrite', True)

    if dfs.ExistFile(settings) and not overwrite:
        raise Exception("File already exists in this source.")

    to_save = pickle.dumps(model, 0)
    success, dfs = dfs.writeBlock(settings, to_save, None, False)
    return [success]


@task(filename=FILE_OUT)
def save_model_fs(model, filename):
    """SaveModel.

    Save a machine learning model into a file.
    :param filename: Absolute path of the file;
    :param model: The model to be saved
    """
    with open(filename, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


@task(returns=1)
def load_model_hdfs(settings):
    """LoadModelFromHDFS.

    Load a machine learning model from a HDFS source.
    :param settings: A dictionary with:
        - path:      The path of the file from the / of the HDFS;
        - host:      The host of the Namenode HDFS; (default, 'default')
        - port:      Port of the Namenode HDFS; (default, 0)
    :return:         Returns a model (a dictionary)
    """
    from hdfspycompss.HDFS import HDFS
    from hdfspycompss.Block import Block
    host = settings.get('host', 'localhost')
    port = settings.get('port', 9000)
    filename = settings['filename']

    dfs = HDFS(host=host, port=port)
    blk = dfs.findNBlocks(filename, 1)
    to_load = Block(blk).readBinary()
    model = None
    if len(to_load) > 0:
        model = pickle.loads(to_load)
    return model


@task(returns=1, filename=FILE_IN)
def load_model_fs(filename):
    """LoadModel.

    Load a machine learning model from a file.
    :param filename: Absolute path of the file;
    :return:         Returns a model (a dictionary).
    """
    with open(filename, 'rb') as input:
        model = pickle.load(input)
    return model
