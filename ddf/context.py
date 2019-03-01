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

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.functions.reduce import merge_reduce
from collections import OrderedDict, deque
import ddf
import copy

DEBUG = False


class COMPSsContext(object):
    """
    Controls the DDF tasks executions
    """
    adj_tasks = dict()
    schemas_map = dict()
    tasks_map = OrderedDict()
    """
    task_map: a dictionary to stores all following information about a task:

     - name: task name;
     - status: WAIT or COMPLETED;
     - parent: a list with its parents uuid;
     - output: number of output;
     - input: number of input;
     - lazy: Currently, only True or False. True if this task could be 
       grouped with others in a COMPSs task;
     - function: 
       - if status is WAIT: a list with the function and its parameters;
       - if status is COMPLETED: a dictionary with the results. The keys of 
         this dictionary is index that represents the output (to handle with 
         multiple outputs, like split task);
     - n_input: a ordered list that informs the id key of its parent output
    """

    @staticmethod
    def get_var_by_task(variables, uuid):
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

    def run_workflow(self, wanted=-1):
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
            if isinstance(obj, ddf.DDF):
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
                n_input = self.tasks_map[child_task].get('n_input', [0])

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
                        # if id_in == -1:
                        #     id_in = 0
                        #     print "NUNCA SERA EXCECUTADA"
                        inputs[d] = self.tasks_map[id_p].get('function',
                                                             [0])[id_in]
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

                        self.schemas_map[child_task] = self.schemas_map[id_p]

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

                    # non lazy tasks that need to be executed separated
                    elif not self.tasks_map[child_task].get('lazy', False):
                        self.run_non_lazy_tasks(child_task, variables, id_var,
                                                id_parents, n_input)

                    elif self.tasks_map[child_task]['lazy']:
                        self.run_serial_lazy_tasks(i_task, child_task,
                                                   tasks_list,
                                                   tasks_to_in_count, id_var,
                                                   variables)

    def run_non_lazy_tasks(self, child_task, variables, id_var,
                           id_parents, n_input):
        # Execute f and put result in variables
        if DEBUG:
            print "RUNNING {} ({}) - condition 3.".format(
                    self.tasks_map[child_task]['name'],
                    child_task)

        # get the operation to be executed
        operation = self.tasks_map[child_task]['function']

        # some operations need a prior information
        if self.tasks_map[child_task].get('info', False):
            operation[1]['info'] = []
            for p, i in zip(id_parents, n_input):
                sc = self.schemas_map[p][i]
                operation[1]['info'].append(sc)

        # execute this operation that returns a dictionary
        output_dict = variables[id_var]._execute_task(operation)
        keys_data = output_dict['key_data']
        keys_info = output_dict['key_info']
        info = [merge_reduce(merge_schema, output_dict[f]) for f in keys_info]

        # convert the output and schema to the right format {0: ... 1: ....}
        out_data = {}
        out_info = {}
        for f, key in enumerate(keys_data):
            out_data[f] = output_dict[key]
            out_info[f] = info[f]

        # save results in task_map and schemas_map
        self.tasks_map[child_task]['function'] = out_data
        self.tasks_map[child_task]['status'] = 'COMPLETED'
        self.schemas_map[child_task] = out_info

    def run_serial_lazy_tasks(self, i_task, child_task, tasks_list,
                              tasks_to_in_count, id_var, variables):
        opt_uuids = set()
        opt_functions = []

        if DEBUG:
            print "RUNNING {} ({}) - condition 4.".format(
                    self.tasks_map[child_task]['name'], child_task)

        for id_j, task_opt in enumerate(tasks_list[i_task:]):
            if DEBUG:
                print 'Checking lazziness: {} -> {}'.format(
                        task_opt[:8],
                        self.tasks_map[task_opt]['name'])

            opt_uuids.add(task_opt)
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
            print "Stages (optimized): {}".format(opt_uuids)
            print "opt_functions", opt_functions
        result, info = variables[id_var]._execute_lazy(opt_functions)

        out_data = {}
        out_info = {}

        for o in opt_uuids:
            self.tasks_map[o]['status'] = 'COMPLETED'

            # output format: {0: ... 1: ....}
            n_outputs = self.tasks_map[o]['output']

            if n_outputs == 1:
                out_data[0] = result
                out_info[0] = info
            else:
                for f in range(n_outputs):
                    out_data[f] = result[f]
                    out_info[f] = info[f]

            self.tasks_map[o]['function'] = out_data
            self.schemas_map[o] = out_info

            if DEBUG:
                print "{} ({}) is COMPLETED - condition 4." \
                    .format(self.tasks_map[o]['name'], o[:8])


@task(returns=2)
def task_bundle(data, stage, id_frag):
    info = []
    for f, task in enumerate(stage):
        function, settings = task
        # Used only in save
        if isinstance(settings, dict):
            settings['id_frag'] = id_frag
        t = function(data, settings)
        print f
        data, info = t

    return data, info


@task(returns=1)
def merge_schema(schema1, schema2):

    columns1, dtypes1, p1 = schema1
    columns2, dtypes2, p2 = schema2

    schema = [columns1, dtypes1, p1 + p2]
    return schema
