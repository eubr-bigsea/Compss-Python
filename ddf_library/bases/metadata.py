#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from prettytable import PrettyTable

from ddf_library.utils import _gen_uuid
from ddf_library.utils import check_serialization, delete_result
import networkx as nx


class Status(object):

    STATUS_WAIT = 'WAIT'
    STATUS_DELETED = 'DELETED'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_PERSISTED = 'PERSISTED'  # to persist in order to reuse later


class OPTGroup(object):

    OPT_SERIAL = 'serial'  # it can be grouped with others operations
    OPT_OTHER = 'other'  # it can not be performed any kind of task optimization
    OPT_LAST = 'last'  # it contains two or more stages,
    # but only the last stage can be grouped


class CatalogTask(object):

    dag = nx.DiGraph()

    task_definitions = dict()
    # A dictionary with static information about some operations:
    #  - output: number of output;
    #  - input: number of input;
    #  - optimization: Currently, 'serial', 'other' or 'last'.
    #  - info_condition: If task needs a parent schema;  # TODO

    catalog_tasks = dict()
    # task_map: a dictionary to stores all following information about a task:
    #      - name: task name
    #      - function: a list with the function and its parameters;
    #      - status: WAIT, COMPLETED, PERSISTED
    #      - result: if status is COMPLETED, a dictionary with the results.
    #         The keys of this dictionary is index that represents the output
    #         (to handle with multiple outputs, like split task);
    #      - schema:

    # to speedup the searching for completed tasks:
    completed_tasks = list()

    def add_definition(self, name, args):
        if name not in self.task_definitions:
            self.task_definitions[name] = args

    def clear(self):  # TODO

        for id_task in list(self.catalog_tasks):
            data = self.get_task_return(id_task)

            if check_serialization(data):
                delete_result(data)

        self.task_definitions = dict()
        self.catalog_tasks = dict()
        self.completed_tasks = list()
        self.dag = nx.DiGraph()

    def show_tasks(self):
        """
        Show all tasks in the current code. Only to debug.
        :return:
        """

        t = PrettyTable(['uuid', 'Task name', 'STATUS', 'Example result'])
        for uuid in self.catalog_tasks:
            r = self.get_task_return(uuid)
            if isinstance(r, list):
                if len(r) > 0:
                    r = r[0]
                else:
                    r = ''
            t.add_row([uuid[:8], self.get_task_name(uuid),
                       self.get_task_status(uuid), r])
        print("\nList of all tasks:\n", t, '\n')

    def gen_new_uuid(self):
        new_state_uuid = _gen_uuid()
        while new_state_uuid in self.catalog_tasks:
            new_state_uuid = _gen_uuid()
        return new_state_uuid

    def add_completed(self, uuid):
        self.completed_tasks.append(uuid)

    def rm_completed(self, uuid):
        # to prevent multiple occurrences
        self.completed_tasks = \
            list(filter(lambda a: a != uuid,
                        self.completed_tasks))

    def list_completed(self):
        return self.completed_tasks

    def list_all(self):
        return list(self.catalog_tasks)

    def get_schema(self, uuid):
        return self.catalog_tasks[uuid].get('schema', None)

    def set_schema(self, uuid, schema):
        self.catalog_tasks[uuid]['schema'] = schema

    def rm_schema(self, uuid):
        self.catalog_tasks[uuid].pop('schema', None)

    def set_new_task(self, uuid_task, args):
        self.catalog_tasks[uuid_task] = args

    def get_task_name(self, uuid_task):
        return self.catalog_tasks[uuid_task].get('name', '')

    def get_task_opt_type(self, uuid_task):
        name_task = self.get_task_name(uuid_task)
        return self.task_definitions[name_task].get('optimization',
                                                    OPTGroup.OPT_OTHER)

    def get_task_function(self, uuid_task):
        return self.catalog_tasks[uuid_task]['function']

    def set_task_function(self, uuid_task, function):
        self.catalog_tasks[uuid_task]['function'] = function

    def get_task_function2(self, uuid_task):  # TODO: rename it
        return self.catalog_tasks[uuid_task]['function'][0]

    def set_task_function2(self, uuid_task, function):
        self.catalog_tasks[uuid_task]['function'][0] = function

    def get_task_parameters(self, uuid_task):
        return self.catalog_tasks[uuid_task]['function'][1]

    def set_task_parameters(self, uuid_task, params):
        self.catalog_tasks[uuid_task]['function'][1] = params

    def get_task_return(self, uuid_task):
        return self.catalog_tasks[uuid_task].get('result', [])

    def set_task_result(self, uuid_task, data):
        self.catalog_tasks[uuid_task]['result'] = data

    def get_task_status(self, uuid_task):
        return self.catalog_tasks[uuid_task].get('status', Status.STATUS_WAIT)

    def set_task_status(self, uuid_task, status):
        self.catalog_tasks[uuid_task]['status'] = status
        if status == Status.STATUS_COMPLETED:
            self.completed_tasks.append(uuid_task)

    def get_task_parents(self, uuid_task):
        return [i for i, o in self.dag.in_edges(uuid_task)]

    def get_n_input(self, uuid_task):
        name_task = self.get_task_name(uuid_task)
        return self.task_definitions[name_task]['input']

    def get_task_sibling(self, uuid_task):
        return self.catalog_tasks[uuid_task].get('sibling', [uuid_task])

    def set_task_sibling(self, uuid_task, siblings):
        if uuid_task not in self.catalog_tasks:
            raise Exception('uuid "{}" not in '
                            'catalog_tasks'.format(uuid_task[:8]))
        self.catalog_tasks[uuid_task]['sibling'] = siblings

    def get_input_data(self, id_parents):
        return [self.get_task_return(id_p) for id_p in id_parents]

    def get_info_condition(self, uuid_task):
        name_task = self.get_task_name(uuid_task)
        return self.task_definitions[name_task].get('info', False)
