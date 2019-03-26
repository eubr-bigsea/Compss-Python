#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.


"""

from pycompss.api.task import task
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

    def show_workflow(self, selected_tasks):
        """
        Show the final workflow. Only to debug
        :param selected_tasks: list of tasks to be executed in this flow.
        """
        print("Relevant tasks:")
        for uuid in selected_tasks:
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

            while enter:
                dfs(enter.pop())
            return order

        for k in self.adj_tasks:
            self.adj_tasks[k] = list(set(self.adj_tasks[k]))
            if DEBUG:
                print "{} ({}) --> {}".format(k, self.tasks_map[k]['name'],
                                              self.adj_tasks[k])

        result = list(topological(self.adj_tasks))

        return result

    def get_taskslist(self, wanted):
        import gc
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
        selected_tasks = list()
        for var in variables:
            selected_tasks.extend(var.task_list)

        return selected_tasks, variables

    def get_task_name(self, uuid_task):
        return self.tasks_map[uuid_task]['name']

    def get_task_lazyness(self, uuid_task):
        return self.tasks_map[uuid_task].get('lazy', False)

    def get_task_function(self, uuid_task, id_in=None):
        if id_in is not None:
            return self.tasks_map[uuid_task].get('function', [0])[id_in]
        return self.tasks_map[uuid_task]['function']

    def set_task_function(self, uuid_task, data, id_in=None):
        if id_in is not None:
            self.tasks_map[uuid_task]['function'][id_in] = data
        else:
            self.tasks_map[uuid_task]['function'] = data

    def get_task_status(self, uuid_task):
        return self.tasks_map[uuid_task]['status']

    def set_task_status(self, uuid_task, status):
        self.tasks_map[uuid_task]['status'] = status

    def get_task_parents(self, uuid_task):
        return self.tasks_map[uuid_task]['parent']

    def get_task_ninput(self, uuid_task):
        return self.tasks_map[uuid_task].get('n_input', [0])

    def set_auxiliary_variables(self, variables, current_task):
        id_var = self.get_var_by_task(variables, current_task)
        if len(id_var) == 0:
            raise Exception("\nVariable was deleted")

        id_var = id_var[0]
        tasks_list = variables[id_var].task_list
        id_task = tasks_list.index(current_task)
        tasks_list = tasks_list[id_task:]

        return id_var, tasks_list

    def set_input(self, id_parents, n_input):

        inputs = {}
        for d, (id_p, id_in) in enumerate(zip(id_parents, n_input)):
            inputs[d] = self.get_task_function(id_p, id_in)

        return inputs

    def is_ready_to_run(self, id_parents):
        return all([self.get_task_status(p) == 'COMPLETED' for p in id_parents])

    def set_operation(self, child_task, id_parents, n_input):

        # get the operation to be executed
        operation = self.get_task_function(child_task)

        # some operations need a prior information
        if self.tasks_map[child_task].get('info', False):
            operation[1]['info'] = []
            for p, i in zip(id_parents, n_input):
                sc = self.schemas_map[p][i]
                if not isinstance(sc[0], list):
                    sc = merge_reduce(merge_schema, sc)
                operation[1]['info'].append(sc)

        return operation

    def run_workflow(self, wanted=-1):
        """
        Find flow of tasks non executed with an action (show, save, cache, etc)
        and execute each flow until a 'sync' is found. Action tasks represents
        operations which its results will be saw by the user.

        :param wanted: uuid of an specific task
        """

        # list all tasks that must be executed and its ddf variables
        selected_tasks, ddf_variables = self.get_taskslist(wanted)
        # and perform a topological sort to create a DAG
        topological_tasks = self.create_adj_tasks(selected_tasks)

        if DEBUG:
            self.show_tasks()
            self.show_workflow(selected_tasks)
            print "topological_tasks: ", topological_tasks

        # iterate over all sorted tasks
        for current_task in topological_tasks:

            if self.get_task_status(current_task) == 'COMPLETED':
                continue

            # get its variable and related tasks
            id_var, tasks_list = self.set_auxiliary_variables(ddf_variables,
                                                              current_task)
            for i_task, child_task in enumerate(tasks_list):

                if self.get_task_status(child_task) == 'COMPLETED':
                    continue

                id_parents = self.get_task_parents(child_task)
                n_input = self.get_task_ninput(child_task)

                if DEBUG:
                    print(" - task {} ({})".format(
                            self.get_task_name(child_task), child_task[:8]))
                    print "id_parents: {} and n_input: {}".format(
                            id_parents, n_input)

                # when has parents: wait all parents tasks be completed
                if not self.is_ready_to_run(id_parents):
                    break

                # get input data from parents
                inputs = self.set_input(id_parents, n_input)

                # end this path
                if self.get_task_name(child_task) == 'sync':
                    self.run_sync(child_task, id_parents, n_input, inputs)
                    break

                # non lazy tasks that need to be executed separated
                elif not self.get_task_lazyness(child_task):
                    self.run_non_lazy_tasks(child_task, id_parents,
                                            n_input, inputs)

                # lazy tasks that could be executed in group
                elif self.get_task_lazyness(child_task):
                    self.run_serial_lazy_tasks(i_task, child_task, tasks_list,
                                               selected_tasks, inputs)

    def run_sync(self, child_task, id_parents, n_input, inputs):
        """
        Execute the sync task, in this context, it means that all COMPSs tasks
        until this current sync will be executed. This do not
        represent a COMPSs synchronization.
        """
        if DEBUG:
            print "RUNNING sync ({}) - condition 2." \
                .format(child_task)

        # sync tasks always will have only one parent
        id_p = id_parents[0]
        id_in = n_input[0]
        result = inputs[0]

        self.set_task_function(child_task, self.get_task_function(id_p))
        self.schemas_map[child_task] = self.schemas_map[id_p]

        self.set_task_function(id_p, result, id_in)
        self.set_task_function(child_task, result, id_in)

        self.set_task_status(child_task, 'COMPLETED')
        self.set_task_status(id_p, 'COMPLETED')

    def run_non_lazy_tasks(self, child_task, id_parents, n_input, inputs):
        """
        The current operation can not be grouped with other operations, so,
        it must be executed separated.
        """

        if DEBUG:
            print "RUNNING {} ({}) - condition 3.".format(
                    self.tasks_map[child_task]['name'],
                    child_task)

        operation = self.set_operation(child_task, id_parents, n_input)

        # execute this operation that returns a dictionary
        output_dict = self._execute_task(operation, inputs)

        self.save_non_lazy_states(output_dict, child_task)

    def run_serial_lazy_tasks(self, i_task, child_task, tasks_list,
                              selected_tasks, inputs):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = set(), list()

        if DEBUG:
            print "RUNNING {} ({}) - condition 4.".format(
                    self.tasks_map[child_task]['name'], child_task)

        for id_j, task_opt in enumerate(tasks_list[i_task:]):
            if DEBUG:
                print 'Checking lazziness: {} -> {}'.format(
                        task_opt[:8],
                        self.tasks_map[task_opt]['name'])

            group_uuids.add(task_opt)
            group_func.append(self.get_task_function(task_opt))

            if (i_task + id_j + 1) < len(tasks_list):
                next_task = tasks_list[i_task + id_j + 1]

                if selected_tasks.count(task_opt) != \
                        selected_tasks.count(next_task):
                    break

                if not all([self.get_task_lazyness(task_opt),
                            self.get_task_lazyness(next_task)]):
                    break

        if DEBUG:
            print "Stages (optimized): {}".format(group_uuids)
            print "opt_functions", group_func

        result, info = self._execute_lazy(group_func, inputs)
        self.save_lazy_states(result, info, group_uuids)

    def save_non_lazy_states(self, output_dict, child_task):
        # Results in non lazy tasks are in dictionary format

        keys_data = output_dict['key_data']
        keys_info = output_dict['key_info']
        info = [output_dict[f] for f in keys_info]

        # convert the output and schema to the right format {0: ... 1: ....}
        out_data, out_info = dict(), dict()
        for f, key in enumerate(keys_data):
            out_data[f] = output_dict[key]
            out_info[f] = info[f]

        # save results in task_map and schemas_map
        self.schemas_map[child_task] = out_info
        self.set_task_function(child_task, out_data)
        self.set_task_status(child_task, 'COMPLETED')

    def save_lazy_states(self, result, info, opt_uuids):
        out_data, out_info = dict(), dict()
        for o in opt_uuids:

            # output format: {0: ... 1: ....}
            n_outputs = self.tasks_map[o]['output']

            if n_outputs == 1:
                out_data[0] = result
                out_info[0] = info
            else:
                for f in range(n_outputs):
                    out_data[f] = result[f]
                    out_info[f] = info[f]

            self.set_task_function(o, out_data)
            self.schemas_map[o] = out_info
            self.set_task_status(o, 'COMPLETED')

            if DEBUG:
                print "{} ({}) is COMPLETED - condition 4." \
                    .format(self.tasks_map[o]['name'], o[:8])

    @staticmethod
    def _execute_task(env, input_data):
        """
        Used to execute all non-lazy functions.

        :param f: a list that contains the current task and its parameters.
        :return:
        """

        function, settings = env
        if len(input_data) > 1:
            partitions = [input_data[k] for k in input_data]
        else:
            partitions = input_data[0]

        output = function(partitions, settings)
        return output

    @staticmethod
    def _execute_lazy(opt, data):

        """
        Used to execute a group of lazy tasks. This method submit
        multiple 'context.task_bundle', one for each data fragment.

        :param opt: sequence of functions and parameters to be executed in
            each fragment
        :param data: input data
        :return:
        """

        tmp = None
        if len(data) > 1:
            tmp = [data[k] for k in data]
        else:
            for k in data:
                tmp = data[k]

        result, info = [[] for _ in tmp], [[] for _ in tmp]
        for f, df in enumerate(tmp):
            result[f], info[f] = task_bundle(df, opt, f)

        return result, info


@task(returns=2)
def task_bundle(data, stage, id_frag):
    info = []
    for f, current_task in enumerate(stage):
        function, settings = current_task
        if isinstance(settings, dict):
            # Used only in save
            settings['id_frag'] = id_frag
        data, info = function(data, settings)

    return data, info


@task(returns=1)
def merge_schema(schema1, schema2):

    columns1, dtypes1, p1 = schema1
    columns2, dtypes2, p2 = schema2

    schema = [columns1, dtypes1, p1 + p2]
    return schema
