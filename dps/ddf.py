#!/usr/bin/python

import itertools

import os
from collections import defaultdict, deque, OrderedDict

import uuid
from pycompss.api.api import compss_barrier, compss_open
from tasks import *
import copy
from  heapq3 import  *

import pandas as pd


class COMPSsContext(object):
    tasks_map = OrderedDict()  # id: {EXECUTED, result}

    adj_tasks = dict()

    def get_var_by_task(self, vars, uuid):
        """
        Return a variable id which contains the task uuid.

        :param vars:
        :param uuid:
        :return:
        """
        return [i for i, v in enumerate(vars) if uuid in v.task_list]

    def show_workflow(self, list_uuid):
        """
        Show the final workflow. Only to debug
        :param list_uuid:
        :return:
        """
        print "[LOG] - Tasks to take in count:"
        for uuid in list_uuid:
            print "\t{} - ({})".format(self.tasks_map[uuid]['name'], uuid[:8])
        print "-" * 20

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



        # for t in self.tasks_map:
        #     parents = self.tasks_map[t]['parent']
        #     for p in parents:
        #         if p not in self.adj_tasks:
        #             self.adj_tasks[p] = []
        #         self.adj_tasks[p].append(t)

        from collections import deque

        GRAY, BLACK = 0, 1

        def topological(graph):
            order, enter, state = deque(), set(graph), {}

            def dfs(node):
                state[node] = GRAY
                for k in graph.get(node, ()):
                    sk = state.get(k, None)
                    if sk == GRAY: raise ValueError("cycle")
                    if sk == BLACK: continue
                    enter.discard(k)
                    dfs(k)
                order.appendleft(node)
                state[node] = BLACK

            while enter: dfs(enter.pop())
            return order

        result = list(topological(self.adj_tasks))
        print "TOPOLOGICAL SORT:", result
        print "-" * 20
        return result


    def is_ready(self, uuid_task):
        pass


    def run(self):
        import gc

        print "map:"
        for k in self.tasks_map:
            print "{}: {}".format(k[:8], self.tasks_map[k])
        print '\n\n'

        action_tasks = []
        for t in self.tasks_map:
            if self.tasks_map[t]['name'] in ['save', 'sync']:
                action_tasks.append(t)

        all_variables = []
        final_variables = []  # variaveis finais
        for obj in gc.get_objects():
            if isinstance(obj, DPS):
                all_variables.append(obj)
                tasks = obj.task_list
                for k in action_tasks:
                    if k in tasks:
                        final_variables.append(obj)

        # final_variables = all_variables  #!TODO
        print "[LOG] - Number of variables: ", len(final_variables)
        tasks_to_in_count = list()
        for var in final_variables:
            tasks_to_in_count.extend(var.task_list)



        # Checking for tasks that need for others input

        # tasks_to_in_count = [k for k in self.tasks_map]

        self.show_workflow(tasks_to_in_count)
        topological_tasks = self.create_adj_tasks(tasks_to_in_count)

        for current_task in topological_tasks:
            print "[LOG] - Current task: {} ({})".format(self.tasks_map[current_task]['name'], current_task)

            id_var = self.get_var_by_task(final_variables, current_task)[0]
            tasks_list = final_variables[id_var].task_list
            id_task = tasks_list.index(current_task)

            for i, child_task in enumerate(tasks_list[id_task:]):
                child_task = tasks_list[i]
                print "[LOG] - {}o task {} ({})".format(i+1, self.tasks_map[child_task]['name'], child_task)
                print "[LOG] - {} ".format(self.tasks_map[child_task])

                id_parents = self.tasks_map[child_task]['parent']
                if self.tasks_map[child_task]['status'] == 'WAIT':
                    if not all([self.tasks_map[p]['status'] == 'COMPLETED' for p in id_parents]):
                        print "[LOG] - WAITING FOR A PARENT BE COMPLETED"
                        break
                    # atualiza os dados

                    n_input = self.tasks_map[child_task].get('n_input', [-1])

                    if len(id_parents) > 0:

                        inputs = {}
                        for ii, p in enumerate(id_parents):
                            # se o output do pai for dividido (ex.: split)
                            n_input_current = n_input[ii]
                            if n_input_current != -1:
                                inputs[ii] = self.tasks_map[p]['function'][n_input_current]
                            else:
                                tmp = self.tasks_map[p]['function']
                                if isinstance(tmp, dict):
                                    tmp = tmp[0]
                                inputs[ii] = tmp

                        print "INPUTS:", inputs
                        final_variables[id_var].partitions = inputs


                    # mark the begining
                    if self.tasks_map[child_task]['name'] == 'init':
                        self.tasks_map[child_task]['status'] = 'COMPLETED'
                        print "[LOG] - init ({}) is COMPLETED - condition 1".format(
                            child_task)

                    elif not self.tasks_map[child_task]['lazy']:
                        # Execute f and put result in vars
                        if self.tasks_map[child_task]['name'] == 'sync':
                            final_variables[id_var].action()
                            self.tasks_map[child_task]['function'] = \
                                final_variables[id_task].partitions

                            self.tasks_map[child_task]['status'] = 'COMPLETED'
                            print "sync ({}) is COMPLETED - condition 2.".format(
                                child_task)
                            print self.tasks_map[child_task]

                        elif self.tasks_map[child_task].get('input', 1) < 2:

                            # tarefas com 1 input e que nao podem ser agrupadas
                            print "[LOG] - RUNNING {} ({}) - condition 3".format(
                                    self.tasks_map[child_task]['name'],
                                    child_task)
                            f = self.tasks_map[child_task]['function']
                            final_variables[id_var].task_others(f)
                            self.tasks_map[child_task][
                                'status'] = 'COMPLETED'
                            self.tasks_map[child_task]['function'] = \
                            final_variables[id_var].partitions
                            print '[LOG] - {} ({}) is COMPLETED'.format(
                                    self.tasks_map[child_task]['name'],
                                    child_task)
                            print '[LOG] - {}\n---'.format(
                                    self.tasks_map[child_task])

                        else:
                            #preciso chegar se todos os parents estao COMPLETED
                            print "[LOG] - RUNNING {} ({}) - condition 4".format(
                                    self.tasks_map[child_task]['name'],
                                    child_task)
                            f_task = self.tasks_map[child_task]['function']
                            final_variables[id_var].task_others(f_task)

                            self.tasks_map[child_task][
                                'status'] = 'COMPLETED'
                            self.tasks_map[child_task]['function'] = \
                                final_variables[id_task].partitions

                            self.tasks_map[child_task][
                                'status'] = 'COMPLETED'
                            print '{} ({}) is COMPLETED - condition 4'.format(
                                    self.tasks_map[child_task]['name'],
                                    child_task)

                        pass

                    else:
                        opt = set()
                        opt_functions = []
                        last_function = self.tasks_map[tasks_list[i - 1]][
                            'function']
                        for j in xrange(i, len(tasks_list)):
                            t = tasks_list[j]
                            opt.add(t)
                            opt_functions.append(
                                    self.tasks_map[t]['function'])
                            if tasks_to_in_count.count(
                                    tasks_list[j]) != tasks_to_in_count.count(
                                    tasks_list[j + 1]):
                                break
                            if not all([self.tasks_map[t]['lazy'],
                                        self.tasks_map[tasks_list[j + 1]][
                                            'lazy']]):
                                break

                        print "OPT: ", opt
                        print "OPT:", opt_functions
                        final_variables[id_task].perform(opt_functions,
                                                    last_function)

                        for o in opt:
                            self.tasks_map[o]['status'] = 'COMPLETED'
                            self.tasks_map[o]['function'] = final_variables[
                                id_task].partitions

                            print '{} ({}) is COMPLETED - condition 5'.format(
                                    self.tasks_map[o]['name'], o)
                            print '[LOG] - {}\n---'.format(
                                    self.tasks_map[o])

        print "map:"
        for k in self.tasks_map:
            print "{} --> {}".format(k, self.tasks_map[k])
        print '\n\n'

    def check_brothers(self, b, uuid):
        """
        Check if a and b are brothers (if have both uuid)
        :return:
        """

        return uuid in b.task_list

    def check_lazziness(self, uuid):
        """
        Check if task is lazy

        :param uuid:
        :return:
        """

        return self.tasks_map[uuid]['lazy']

    def check_forward(self, a, b, uuid):
        """
        Check if a and b at task uuid have same next
        :param a:
        :param b:
        :param uuid:
        :return:
        """

        pass



class DPS(object):
    """
    Distributed Data Handler.
    Should distribute the data and run tasks for each partition.
    """

    def __init__(self, partitions=None, task_list=None, last_uuid='init', settings={'input': -1}):
        super(DPS, self).__init__()

        self.schema = list()
        self.opt = OrderedDict()
        self.partial_sizes = list()
        self.settings = settings

        if last_uuid != 'init':
            self.partitions = partitions
            self.task_list = copy.deepcopy(task_list)
            self.task_list.append(last_uuid)

        else:
            self.partitions = list()
            self.task_list = list()
            last_uuid = str(uuid.uuid4())
            COMPSsContext.tasks_map[last_uuid] = {'name': 'init', 'lazy': False,
                                                  'input': 0, 'parent': [],
                                                  'status': 'COMPLETED'}
            self.task_list.append(last_uuid)

        self.last_uuid = last_uuid

        # if isinstance(input, pd.DataFrame):
        #     self.load_df(input, num_of_parts)
        # else:
        #     self.load_fs(input, num_of_parts)

    def load_fs(self, filename, num_of_parts=4):

        from functions.data.read_data import ReadOperationHDFS

        settings = dict()
        settings['port'] = 9000
        settings['host'] = 'localhost'
        settings['separator'] = ','

        self.partitions, info = ReadOperationHDFS()\
            .transform(filename, settings, num_of_parts)

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'load_fs',
                                             'status': 'COMPLETED',
                                             'lazy': False,
                                             'function': self.partitions,
                                             'output': 1, 'input': 0,
                                             'parent': [self.last_uuid]}

        self.partial_sizes.append(info)
        return DPS(self.partitions, self.task_list, uuid_key)

    def task_others(self, f):

        function, settings = f
        tmp = []
        if isinstance(self.partitions, dict):
            if len(self.partitions) > 1:
                for k in self.partitions:
                    tmp.append(self.partitions[k])
            else:
                for k in self.partitions:
                    tmp = self.partitions[k]
        else:
            tmp = self.partitions

        self.partitions = function(tmp, settings)

    def load_df(self, df, num_of_parts=4):
        """
        Use the iterator and create the partitions of this DDS.
        :param iterator:
        :param num_of_parts:
        :return:

        """

        # self.create_partitions(df, num_of_parts)
        from functions.data.data_functions import Partitionize

        self.partitions = Partitionize(df, num_of_parts)

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'load_df',
                                             'status': 'COMPLETED',
                                             'lazy': False,
                                             'function': self.partitions,
                                             'output': 1, 'input': 0,
                                             'parent': [self.last_uuid]}

        return DPS(self.partitions, self.task_list, uuid_key)

    def load_shapefile(self, shp_path, dbf_path, polygon='points',
                       attributes=[], num_of_parts=4):
        """

        :param shp_path: Path to the shapefile (.shp)
        :param dbf_path: Path to the shapefile (.dbf)
        :param polygon: Alias to the new column to store the
                polygon coordenates (default, 'points');
        :param attributes: List of attributes to keep in the dataframe,
                empty to use all fields;
        :return:

        Note: pip install pyshp

        """

        settings = dict()
        settings['shp_path'] = shp_path
        settings['dbf_path'] = dbf_path
        settings['polygon'] = polygon
        settings['attributes'] = attributes

        from functions.geo.read_shapefile import ReadShapeFileOperation

        self.partitions = \
            ReadShapeFileOperation().transform(settings, num_of_parts)

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'load_shapefile',
                                             'status': 'COMPLETED',
                                             'lazy': False,
                                             'function': self.partitions,
                                             'output': 1, 'input': 0,
                                             'parent': [self.last_uuid]}

        return DPS(self.partitions, self.task_list, uuid_key)

    def create_partitions(self, iterator, num_of_parts):
        """
        Saves 'List of future objects' as the partitions. So once called, this
        data set will always contain only future objects.
        :param iterator:
        :param num_of_parts: Number of partitions to be created
                            Should be -1 (minus 1) if iterator is already a list
                            of future objects
        :return:

        >>> DDS().load(range(10), 2).collect(True)
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        """
        if self.partitions:
            raise Exception("Partitions have already been created, cannot load "
                            "new data.")

        if num_of_parts == -1:
            self.partitions = iterator
            return

        total = len(iterator)
        if not total:
            return

        chunk_sizes = [(total // num_of_parts)] * num_of_parts
        extras = total % num_of_parts
        for i in range(extras):
            chunk_sizes[i] += 1

        if isinstance(iterator, basestring):
            start, end = 0, 0
            for size in chunk_sizes:
                end = start + size
                temp = task_load(iterator[start:end])
                start = end
                self.partitions.append(temp)
            return

        start = 0
        for size in chunk_sizes:
            end = start + size
            temp = get_next_partition(iterator, start, end)
            self.partitions.append(temp)
            start = end
        return

    def action(self):

        # uuid_key = str(uuid.uuid4())
        # COMPSsContext.tasks_map[uuid_key] = {'name': 'sync',
        #                                      'status': 'WAIT', 'lazy': False,
        #                                      'parent': [self.last_uuid],
        #                                      'input': 1}
        # task_list = self.task_list

        # for f
        # for f in COMPSsContext.tasks_map:

        self.partitions = compss_wait_on(self.partitions)

    def get_data(self):
        tmp = self.partitions[0]
        return tmp

    def perform(self, opt, data):

        future_objects = []
        for idfrag, p in enumerate(data):
            future_objects.append(task_bundle(p, opt, idfrag))

        self.partitions = future_objects

    def get_tasks(self):
        return self.task_list

    def collect(self, keep_partitions=False, future_objects=False):
        """
        Action

        Returns all elements from all partitions. Elements can be grouped by
        partitions by setting keep_partitions value as True.
        :param keep_partitions: Keep Partitions?
        :param future_objects:
        :return:

        >>> dds = DDS().load(range(10), 2)
        >>> dds.collect(True)
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        >>> DDS().load(range(10), 2).collect()
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """

        # self.action()

        # res = list()
        # # Future objects cannot be extended for now...
        # if future_objects:
        #     return self.partitions
        #
        # self.partitions = compss_wait_on(self.partitions)
        # if not keep_partitions:
        #     res = pd.concat(self.partitions)
        #     # for p in self.partitions:
        #     #     p = compss_wait_on(p)
        #     #     res.extend(p)
        # else:
        #     res = self.partitions
        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'sync',
                                             'status': 'WAIT', 'lazy': False,
                                             'parent': [self.last_uuid],
                                             'function': []}

        self.set_n_input(uuid_key, self.settings['input'])

        return DPS(self.partitions, self.task_list, uuid_key)

    def num_of_partitions(self):
        """
        Get the total amount of partitions
        :return: int

        >>> DDS(range(10), 5).num_of_partitions()
        5
        """
        return len(self.partitions)

    def schema(self):

        return self.schema

    def show(self, n=20, truncate=False):
        """
        Prints the first n rows to the console.

        :param n: Number of rows to show.
        :param truncate: If set to True, truncate strings longer than 20 chars
        by default. If set to a number greater than one, truncates long strings
        to length truncate and align cells right.

        :return:
        """
        res = self.take(n)
        res = compss_wait_on(res.partitions)


        return None

    def toPandas(self):
        """
        Returns the contents of this DataFrame as Pandas pandas.DataFrame.


        :return:
        """
        res = pd.concat(self.get_data())
        return res

    def count(self, reduce=True):
        """
        :return: total number of elements
"
        """

        self.partial_sizes = compss_wait_on(self.partial_sizes)

        if reduce:
            res = sum(self.partial_sizes)
        else:
            res = self.partial_sizes

        return res

    def set_n_input(self, uuid_key, id):

        if 'n_input' not in COMPSsContext.tasks_map[uuid_key]:
            COMPSsContext.tasks_map[uuid_key]['n_input'] = []

        COMPSsContext.tasks_map[uuid_key]['n_input'].append(id)

    def with_column(self, old_column, new_column=None, cast=None):
        """
        Rename or change the data's type of some columns.

        Lazy function

        :param old_column:
        :param new_column:
        :param cast:
        :return:
        """

        from functions.data.attributes_changer import AttributesChangerOperation

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'with_column',
                                             'status': 'WAIT', 'lazy': True,
                                             'function': [task_with_column,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])

        return DPS(self.partitions, self.task_list, uuid_key)

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'aggregation',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_aggregation,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def clean_missing(self, columns, mode='REMOVE_ROW', value=None):

        from functions.etl.clean_missing import CleanMissingOperation

        settings = dict()
        settings['attributes'] = columns
        settings['cleaning_mode'] = mode
        settings['value'] = value

        def task_clean_missing(df, params):
            return CleanMissingOperation().transform(df, params, len(df))

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'clean_missing',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_clean_missing,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def difference(self, data2):
        """
        Returns a new set with containing rows in the first frame but not
        in the second one.

        :param cols:
        :return:
        """
        from functions.etl.difference import DifferenceOperation

        def task_difference(df, params):
            return DifferenceOperation()\
                .transform(df[0], df[1], len(df[0]))

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'difference',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_difference, {}],
                                             'parent': [self.last_uuid,
                                                        data2.last_uuid],
                                             'output': 1, 'input': 2}

        self.set_n_input(uuid_key, self.settings['input'])
        self.set_n_input(uuid_key, data2.settings['input'])
        return DPS(self.partitions, self.task_list+ data2.task_list, uuid_key)

    def distinct(self, cols):
        from functions.etl.distinct import DistinctOperation

        def task_distinct(df, params):
            return DistinctOperation().transform(df, params, len(df))

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'distinct',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_distinct, cols],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def drop(self, columns):
        """
        Perform a partial drop operation.
        Lazy function

        :param columns:
        :return:
        """

        def task_drop(df, cols):
            return df.drop(cols, axis=1)

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'drop', 'status': 'WAIT',
                                             'lazy': True,
                                             'function': [task_drop, columns],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def filter(self, expr):
        """
        Filter elements of this data set.

        Lazy function
        :param query: A filtering function
        :return:

        >>> DDS(range(10), 5).filter('col1 == 4').count()
        5
        """

        def task_filter(df, query):
            return df.query(query)

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'filter', 'status': 'WAIT',
                                             'lazy': True,
                                             'function': [task_filter, expr],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

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

        from functions.geo.geo_within import GeoWithinOperation

        settings = dict()
        settings['lat_col'] = lat_col
        settings['lon_col'] = lon_col
        settings['attributes'] = attributes
        settings['polygon'] = polygon
        settings['alias'] = alias

        def task_geo_within(df, params):
            print "task_geo", df[1]
            return GeoWithinOperation().transform(df[0], df[1], params)

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'geo_within',
                                             'status': 'WAIT',
                                             'lazy': False,
                                             'function': [task_geo_within,
                                                          settings],
                                             'parent': [self.last_uuid,
                                                        shp_object.last_uuid],
                                             'output': 1, 'input': 2}

        self.set_n_input(uuid_key, self.settings['input'])
        self.set_n_input(uuid_key, shp_object.settings['input'])
        return DPS(self.partitions, self.task_list + shp_object.task_list,
                   uuid_key)

    def intersect(self, data2):
        """

        :param data2:
        :return:
        """

        from functions.etl.intersect import IntersectionOperation

        def task_intersect(df, params):
            return IntersectionOperation()\
                .transform(df[0], df[1], len(df[0]))

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'intersect',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_intersect, {}],
                                             'parent': [self.last_uuid,
                                                        data2.last_uuid],
                                             'output': 1, 'input': 2}

        self.set_n_input(uuid_key, self.settings['input'])
        self.set_n_input(uuid_key, data2.settings['input'])
        return DPS(self.partitions, self.task_list + data2.task_list, uuid_key)

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'join',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_join, settings],
                                             'parent': [self.last_uuid,
                                                        data2.last_uuid],
                                             'output': 1, 'input': 2}

        self.set_n_input(uuid_key, self.settings['input'])
        self.set_n_input(uuid_key, data2.settings['input'])
        return DPS(self.partitions, self.task_list+ data2.task_list, uuid_key)

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'replace',
                                             'status': 'WAIT', 'lazy': True,
                                             'function': [task_replace,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def sample(self, value=None, seed=None):
        """SampleOperation.

        :param data: A list with nfrag pandas's dataframe;
        :param params: dictionary that contains:
            - type:
                * 'percent': Sample a random amount of records (default)
                * 'value': Sample a N random records
                * 'head': Sample the N firsts records of the dataframe -> take
            - seed : Optional, seed for the random operation.
            - int_value: Value N to be sampled (in 'value' or 'head' type)
            - per_value: Percentage to be sampled (in 'value' or 'head' type)
        :param nfrag: The number of fragments;
        :return: A list with nfrag pandas's dataframe.
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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'sample',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_sample,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1,
                                             }

        self.set_n_input(uuid_key, self.settings['input'])

        return DPS(self.partitions, self.task_list, uuid_key)

    def save(self, filename, format='csv', storage='hdfs',
                  header=True, mode='overwrite'):

        from functions.data.save_data import SaveOperation

        settings = dict()
        settings['filename'] = filename
        settings['format'] = format
        settings['storage'] = storage
        settings['header'] = header
        settings['mode'] = mode

        settings = SaveOperation().preprocessing(settings, len(self.partitions))

        def task_save(df, params):
            return SaveOperation().transform_serial(df, params)

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'save', 'status': 'WAIT',
                                             'lazy': True,
                                             'function': [task_save, settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def select(self, columns):
        """
        Perform a partial projection.

        Lazy function
        :param columns:
        :return:
        """

        def task_select(df, fields):
            # remove the columns that not in list1
            fields = [field for field in fields if field in df.columns]
            if len(fields) == 0:
                raise Exception("The columns passed as parameters "
                                "do not belong to this DataFrame.")
            return df[fields]

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'select', 'status': 'WAIT',
                                             'lazy': True,
                                             'function': [task_select, columns],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def sort(self, cols,  ascending=[]):

        from functions.etl.sort import SortOperation

        settings = dict()
        settings['columns'] = cols
        settings['ascending'] = ascending

        def task_sort(df, params):
            return SortOperation().transform(df, params, len(df))

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'sort',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_sort,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        return DPS(self.partitions, self.task_list, uuid_key)

    def split(self, percentage=0.5, seed=None):

        from functions.etl.split import SplitOperation

        settings = dict()
        settings['percentage'] = percentage
        settings['seed'] = seed

        def task_split(df, params):
            return SplitOperation().transform(df, params, len(df))

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'split',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_split,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 2, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key, {'input': 0}), \
               DPS(self.partitions, self.task_list, uuid_key, {'input': 1})

    def take(self, num):
        """
        Returns the first num rows as a list of Row.

        :param num:
        :return:
        """

        from functions.etl.sample import SampleOperation
        settings = dict()
        settings['type'] = 'head'
        settings['int_value'] = num

        def task_take(df, params):
            return SampleOperation().transform(df, params, len(df))

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'take',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_take,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def transform(self, f, alias):
        """
        Apply a function to each element of this data set.

        Lazy function
        :param f: A function that will take each element of this data set as a
                  parameter
        :param alias:
        :return:

        """

        settings = {'function': f, 'alias': alias}

        def task_transform(df, params):
            function = params['function']
            new_column = params['alias']

            if len(df) > 0:
                v1s = []
                for _, row in df.iterrows():
                    v1s.append(function(row))
                df[new_column] = v1s
            else:
                df[new_column] = np.nan
            return df

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'transform',
                                             'status': 'WAIT', 'lazy': True,
                                             'function': [task_transform,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DPS(self.partitions, self.task_list, uuid_key)

    def union(self, data2):
        """
        Combine this data set with some other DDS data.

        Nao eh tao simples assim, tem q sincronizar com as colunas

        :param args: Arbitrary amount of DDS objects.
        :return:
        """

        # for dds in args:
        #     self.partitions.extend(dds.partitions)

        from functions.etl.union import UnionOperation

        def task_union(df, params):
            return UnionOperation.transform(df[0], df[2])

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'union',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_union,
                                                          {}],
                                             'parent': [self.last_uuid,
                                                        data2.last_uuid],
                                             'output': 1, 'input': 2}

        self.set_n_input(uuid_key, self.settings['input'])
        self.set_n_input(uuid_key, data2.settings['input'])
        return DPS(self.partitions, self.task_list + data2.task_list, uuid_key)



