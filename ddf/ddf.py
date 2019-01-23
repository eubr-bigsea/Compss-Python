#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from collections import OrderedDict, deque
import uuid

from tasks import *
import copy

import pandas as pd


from collections import OrderedDict, deque

from tasks import *
import copy


class COMPSsContext(object):
    """
    Controls the execution of DDF tasks
    """
    tasks_map = OrderedDict()
    adj_tasks = dict()

    def get_var_by_task(self, vars, uuid):
        """
        Return the variable id which contains the task uuid.

        :param vars:
        :param uuid:
        :return:
        """
        return [i for i, v in enumerate(vars) if uuid in v.task_list]

    def show_workflow(self, tasks_to_in_count):
        """
        Show the final workflow. Only to debug
        :param list_uuid:
        :return:
        """
        print("Tasks to take in count:")
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

        result = list(topological(self.adj_tasks))

        return result

    def run(self, wanted_uuid=None):
        import gc

        print "Called by :", wanted_uuid
        # self.show_tasks()

        # mapping all tasks that produce a final result
        action_tasks = []
        for t in self.tasks_map:
            if self.tasks_map[t]['name'] in ['save', 'sync']:
                action_tasks.append(t)

        # based on that, get the their variables
        vars = []
        n_vars = 0
        for obj in gc.get_objects():
            if isinstance(obj, DDF):
                n_vars += 1
                tasks = obj.task_list
                for k in action_tasks:
                    if k in tasks:
                        vars.append(obj)

        # list all tasks used in these variables
        tasks_to_in_count = list()
        for var in vars:
            tasks_to_in_count.extend(var.task_list)

        # and perform a topological sort to create a DAG
        topological_tasks = self.create_adj_tasks(tasks_to_in_count)
        # print topological_tasks
        # print("Number of variables: {}".format(len(vars)))
        # print("Total of variables: {}".format(n_vars))
        # self.show_workflow(tasks_to_in_count)

        # iterate over all filtered and sorted tasks
        for current_task in topological_tasks:
            print("* Current task: {} ({}) -> {}".format(
                    self.tasks_map[current_task]['name'], current_task[:8],
                    self.tasks_map[current_task]['status']))

            # get its variable and related tasks
            id_var = self.get_var_by_task(vars, current_task)[0]
            tasks_list = vars[id_var].task_list
            id_task = tasks_list.index(current_task)
            print ("* {} em {} ".format(id_task, tasks_list))

            tasks_list = tasks_list[id_task:]
            for i, child_task in enumerate(tasks_list):
                child_task = tasks_list[i]
                id_parents = self.tasks_map[child_task]['parent']
                n_input = self.tasks_map[child_task].get('n_input', [-1])

                print("   - {}o task {} ({}) -> {}"\
                    .format(i+1, self.tasks_map[child_task]['name'],
                            child_task[:8], self.tasks_map[child_task]['status']))

                if self.tasks_map[child_task]['status'] == 'WAIT':
                    print "{}".format(self.tasks_map[child_task])
                    # when has parents: wait all parents tasks be completed
                    if not all([self.tasks_map[p]['status'] == 'COMPLETED'
                                for p in id_parents]):
                        print "WAITING FOR A PARENT BE COMPLETED"
                        break

                    # get the result of each parent
                    if len(id_parents) > 0:
                        inputs = {}
                        for ii, p in enumerate(id_parents):
                            # to handle with parents with multiple outputs
                            n_input_curr = n_input[ii]
                            if n_input_curr != -1:
                                inputs[ii] = \
                                    self.tasks_map[p]['function'][n_input_curr]
                            else:
                                tmp = self.tasks_map[p]['function']
                                if isinstance(tmp, dict):
                                    tmp = tmp[0]
                                inputs[ii] = tmp
                        vars[id_var].partitions = inputs
                        # print "----\n New INPUTS: {}\n".format(inputs)


                    # start the path
                    if self.tasks_map[child_task]['name'] == 'init':
                        self.tasks_map[child_task]['status'] = 'COMPLETED'
                        print "init ({}) is COMPLETED - condition 1"\
                            .format(child_task)

                    # end the path
                    elif self.tasks_map[child_task]['name'] == 'sync':
                        vars[id_var].action()
                        self.tasks_map[child_task]['function'] = \
                            vars[id_var].partitions

                        self.tasks_map[child_task]['status'] = 'COMPLETED'
                        print "sync ({}) is COMPLETED - condition 2."\
                            .format(child_task)

                    elif not self.tasks_map[child_task]['lazy']:
                        # Execute f and put result in vars

                        print "RUNNING {} ({}) - condition 3.".format(
                                self.tasks_map[child_task]['name'], child_task)

                        f = self.tasks_map[child_task]['function']
                        vars[id_var].task_others(f)

                        self.tasks_map[child_task]['status'] = 'COMPLETED'
                        self.tasks_map[child_task]['function'] = \
                            vars[id_var].partitions

                        print "{} ({}) is COMPLETED".format(
                                self.tasks_map[child_task]['name'],
                                child_task)

                    elif self.tasks_map[child_task]['lazy']:
                        opt = set()
                        opt_functions = []
                        # last_function = \
                        #     self.tasks_map[tasks_list[i-1]]['function']

                        for id_j, task_opt in enumerate(tasks_list[i:]):
                            # print 'Checking lazziness: {} --> {}'.format(
                            #         self.tasks_map[task_opt]['name'],
                            #         self.tasks_map[task_opt])


                            opt.add(task_opt)
                            opt_functions.append(
                                    self.tasks_map[task_opt]['function'])

                            if id_j + 1 < len(tasks_list):
                                if tasks_to_in_count.count(task_opt) != \
                                        tasks_to_in_count.count(
                                        tasks_list[id_j + 1]):
                                    print "exit 1"
                                    break

                                if not all([self.tasks_map[task_opt]['lazy'],
                                          self.tasks_map[tasks_list[id_j + 1]][
                                            'lazy']]):
                                    print "exit 2"
                                    break

                        print "Stages (optimized): {}".format(opt)
                        print "opt_functions", opt_functions
                        # print "last:", last_function
                        # print "data", vars[id_task].partitions
                        vars[id_var].perform(opt_functions)

                        for o in opt:
                            self.tasks_map[o]['status'] = 'COMPLETED'
                            self.tasks_map[o]['function'] = \
                                vars[id_var].partitions

                            print "{} ({}) is COMPLETED - condition 4."\
                                .format(self.tasks_map[o]['name'], o[:8])
                            # print self.tasks_map[o]


class DDF(object):
    """
    Distributed DataFrame Handler.
    Should distribute the data and run tasks for each partition.
    """

    def __init__(self, partitions=None, task_list=None,
                 last_uuid='init', settings={'input': -1}):
        super(DDF, self).__init__()

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

    def load_fs(self, filename, num_of_parts=4, header=True, sep=','):

        from functions.data.read_data import ReadOperationHDFS

        settings = dict()
        settings['port'] = 9000
        settings['host'] = 'localhost'
        settings['separator'] = ','
        settings['header'] = header
        settings['separator'] = sep

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
        return DDF(self.partitions, self.task_list, uuid_key)

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

        return DDF(self.partitions, self.task_list, uuid_key)

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

        return DDF(self.partitions, self.task_list, uuid_key)

    def action(self):

        self.partitions = compss_wait_on(self.partitions)

    def get_data(self):
        tmp = self.partitions[0]
        return tmp

    def perform(self, opt):

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

        # self.action()


    def get_tasks(self):
        return self.task_list

    def collect(self):
        """
        Action

        Returns all elements from all partitions. Elements can be grouped by
        partitions by setting keep_partitions value as True.
        :param keep_partitions: Keep Partitions?
        :param future_objects:
        :return:

        """
        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'sync',
                                             'status': 'WAIT', 'lazy': False,
                                             'parent': [self.last_uuid],
                                             'function': []}

        self.set_n_input(uuid_key, self.settings['input'])
        self.task_list.append(uuid_key)
        COMPSsContext().run(self.last_uuid)

        COMPSsContext.tasks_map[uuid_key]['function'] = self.partitions
        return DDF(self.partitions, self.task_list, uuid_key)

    def num_of_partitions(self):
        """
        Get the total amount of partitions
        :return: int

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

        res = pd.concat(self.partitions[0])
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

        return DDF(self.partitions, self.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list+ data2.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list, uuid_key)

    def filter(self, expr):
        """
        Filter elements of this data set.

        Lazy function
        :param query: A filtering function
        :return:

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
        return DDF(self.partitions, self.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list + shp_object.task_list,
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
        return DDF(self.partitions, self.task_list + data2.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list+ data2.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list, uuid_key)

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'sample',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_sample,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1,
                                             }

        self.set_n_input(uuid_key, self.settings['input'])

        return DDF(self.partitions, self.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list, uuid_key)

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'select', 'status': 'WAIT',
                                             'lazy': True,
                                             'function': [task_select, columns],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DDF(self.partitions, self.task_list, uuid_key)

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'sort',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_sort,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        return DDF(self.partitions, self.task_list, uuid_key)

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'split',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_split,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 2, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DDF(self.partitions, self.task_list, uuid_key, {'input': 0}), \
               DDF(self.partitions, self.task_list, uuid_key, {'input': 1})

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

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'take',
                                             'status': 'WAIT', 'lazy': False,
                                             'function': [task_take,
                                                          settings],
                                             'parent': [self.last_uuid],
                                             'output': 1, 'input': 1}

        self.set_n_input(uuid_key, self.settings['input'])
        return DDF(self.partitions, self.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list, uuid_key)

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
        return DDF(self.partitions, self.task_list + data2.task_list, uuid_key)



