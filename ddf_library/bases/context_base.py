#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""

from ddf_library.utils import merge_info
from pycompss.api.api import compss_wait_on

import time
import networkx as nx
from prettytable import PrettyTable


class CONTEXTBASE(object):
    """
    Controls the DDF tasks executions
    """
    catalog = dict()
    tasks_map = dict()
    dag = nx.DiGraph()

    OPT_SERIAL = 'serial'  # it can be grouped with others operations
    OPT_OTHER = 'other'  # it can not be performed any kind of task optimization
    OPT_LAST = 'last'  # it contains two or more stages,
    # but only the last stage can be grouped

    STATUS_WAIT = 'WAIT'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_PERSISTED = 'PERSISTED'
    STATUS_MATERIALIZED = 'MATERIALIZED'  # persisted

    """
    task_map: a dictionary to stores all following information about a task:

     - name: task name;
     - status: WAIT, COMPLETED, TEMP_VIEWED, PERSISTED, MATERIALIZED
     - parent: a list with its parents uuid;
     - output: number of output;
     - input: number of input;
     - optimization: Currently, 'serial', 'other' or 'last'.
     - function: a list with the function and its parameters;
     - result: if status is COMPLETED, a dictionary with the results. The keys 
         of this dictionary is index that represents the output (to handle with 
         multiple outputs, like split task);
     - n_input: a ordered list that informs the id key of its parent output
    """

    @staticmethod
    def show_workflow(tasks_map, selected_tasks):
        """
        Show the final workflow. Only to debug
        :param tasks_map: Context of all tasks;
        :param selected_tasks: list of tasks to be executed in this flow.
        """

        t = PrettyTable(['Order', 'Task name', 'uuid'])
        for i, uuid in enumerate(selected_tasks):
            t.add_row([i+1, tasks_map[uuid]['name'], uuid[:8]])
        print("\nRelevant tasks:")
        print(t)
        print('\n')

    @staticmethod
    def show_tasks(tasks_map):
        """
        Show all tasks in the current code. Only to debug.
        :return:
        """
        print("\nList of all tasks:")

        t = PrettyTable(['uuid', 'Task name'])

        for uuid in tasks_map:
            t.add_row([uuid[:8], tasks_map[uuid]])
        print(t)
        print('\n')

    @staticmethod
    def show_tasks2():
        """
        Show all tasks in the current code. Only to debug.
        :return:
        """
        print("\nList of all tasks:")
        # TODO: each one ?
        t = PrettyTable(['uuid', 'Task name', 'STATUS', 'Result'])

        for uuid in CONTEXTBASE.tasks_map:
            t.add_row([uuid[:8],
                       CONTEXTBASE.tasks_map[uuid]['name'],
                       CONTEXTBASE.tasks_map[uuid]['status'],
                       CONTEXTBASE.tasks_map[uuid].get('result', '')])
        print(t)
        print('\n')

    @staticmethod
    def plot_graph(tasks_map, dag):

        for k, _ in dag.nodes(data=True):
            status = tasks_map[k].get('status', CONTEXTBASE.STATUS_WAIT)
            dag.nodes[k]['style'] = 'filled'
            if dag.nodes[k]['label'] == 'init':
                color = 'black'
                dag.nodes[k]['style'] = 'solid'
            elif status == CONTEXTBASE.STATUS_WAIT:
                color = 'lightgray'
            elif status in [CONTEXTBASE.STATUS_PERSISTED]:
                color = 'forestgreen'
            else:  # temp viewed or completed
                color = 'lightblue'

            dag.nodes[k]['color'] = color

        from networkx.drawing.nx_agraph import write_dot
        t = time.localtime()
        write_dot(dag, 'DAG_{}.dot'.format(time.strftime('%b-%d-%Y_%H%M', t)))

    def create_dag(self, specials=None):
        """
        Create a DAG
        :return: networkx directed Graph
        """

        if specials is None:
            specials = [k for k in self.tasks_map]

        for t in self.tasks_map:
            parents = self.tasks_map[t]['parent']
            if t in specials:
                self.dag.add_node(t, label=self.get_task_name(t))
                for p in parents:
                    if p in specials:
                        self.dag.add_node(p, label=self.get_task_name(p))
                        self.dag.add_edge(p, t)

    def check_action(self, uuid_task):
        return self.tasks_map[uuid_task]['name'] in ['save', 'sync']

    def get_task_name(self, uuid_task):
        return self.tasks_map[uuid_task]['name']

    def get_task_opt_type(self, uuid_task):
        return self.tasks_map[uuid_task].get('optimization', self.OPT_OTHER)

    def get_task_function(self, uuid_task):
        return self.tasks_map[uuid_task]['function']

    def get_task_return(self, uuid_task):
        return self.tasks_map[uuid_task].get('result', [])

    def set_task_function(self, uuid_task, data):
        self.tasks_map[uuid_task]['function'] = data

    def set_task_result(self, uuid_task, data):
        self.tasks_map[uuid_task]['result'] = data

    def get_task_status(self, uuid_task):
        return self.tasks_map[uuid_task]['status']

    def set_task_status(self, uuid_task, status):
        self.tasks_map[uuid_task]['status'] = status

    def get_task_parents(self, uuid_task):
        return self.tasks_map[uuid_task]['parent']

    def get_n_input(self, uuid_task):
        return self.tasks_map[uuid_task]['input']

    def get_task_sibling(self, uuid_task):
        return self.tasks_map[uuid_task].get('sibling', [uuid_task])

    def get_input_data(self, id_parents):
        return [self.get_task_return(id_p) for id_p in id_parents]

    def set_operation(self, child_task, id_parents):

        # get the operation to be executed
        task_and_operation = self.get_task_function(child_task)

        # some operations need a schema information
        if self.tasks_map[child_task].get('info', False):
            task_and_operation[1]['info'] = []
            for p in id_parents:
                sc = self.catalog[p]
                if isinstance(sc, list):
                    sc = merge_info(sc)
                    sc = compss_wait_on(sc)
                    self.catalog[p] = sc
                task_and_operation[1]['info'].append(sc)

        return task_and_operation



