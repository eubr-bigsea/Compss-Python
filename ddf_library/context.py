#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""

from ddf_library.utils import merge_info

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN
from pycompss.api.api import compss_wait_on

from collections import OrderedDict, deque
import copy


DEBUG = False


class COMPSsContext(object):
    """
    Controls the DDF tasks executions
    """
    adj_tasks = dict()
    schemas_map = dict()
    tasks_map = OrderedDict()

    OPT_SERIAL = 'serial'  # it can be grouped with others operations
    OPT_OTHER = 'other'  # it can not be performed any kind of task optimization
    OPT_LAST = 'last'  # it contains two or more stages,
    # but only the last stage can be grouped

    optimization_ops = [OPT_OTHER, OPT_SERIAL, OPT_LAST]
    """
    task_map: a dictionary to stores all following information about a task:

     - name: task name;
     - status: WAIT or COMPLETED;
     - parent: a list with its parents uuid;
     - output: number of output;
     - input: number of input;
     - optimization: Currently, only 'serial' or 'other'. 'serial if this task
       could be grouped with others in a COMPSs task;
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
        print("\n")

    def show_tasks(self):
        """
        Show all tasks in the current code. Only to debug.
        :return:
        """
        print("List of all tasks:")
        for k in self.tasks_map:
            print("{} --> {}".format(k[:8], self.tasks_map[k]))
        print("\n")

    def create_adj_tasks(self, specials=None):
        """
        Create a Adjacency task list
        Parent --> sons
        :return:
        """
        if specials is None:
            specials = []

        for t in self.tasks_map:
            parents = self.tasks_map[t]['parent']
            if t in specials:
                for p in parents:
                    if p in specials:
                        if p not in self.adj_tasks:
                            self.adj_tasks[p] = []
                        self.adj_tasks[p].append(t)

        gray, black = 0, 1

        def topological(graph):
            order, enter, state = deque(), set(graph), {}

            def dfs(node):
                state[node] = gray
                for no in graph.get(node, ()):
                    sk = state.get(no, None)
                    if sk == gray:
                        raise ValueError("cycle")
                    if sk == black:
                        continue
                    enter.discard(k)
                    dfs(k)
                order.appendleft(node)
                state[node] = black

            while enter:
                dfs(enter.pop())
            return order

        for k in self.adj_tasks:
            self.adj_tasks[k] = list(set(self.adj_tasks[k]))
            if DEBUG:
                print("{} ({}) --> {}".format(k, self.tasks_map[k]['name'],
                                              self.adj_tasks[k]))

        result = list(topological(self.adj_tasks))

        return result

    def get_taskslist(self, wanted):
        from ddf_library.ddf import DDF
        import gc
        # mapping all tasks that produce a final result
        action_tasks = []
        if wanted != -1:
            action_tasks.append(wanted)

        for t in self.tasks_map:
            if self.tasks_map[t]['name'] in ['save', 'sync']:
                action_tasks.append(t)

        if DEBUG:
            print("action:", action_tasks)
        # based on that, get the their variables
        variables, n_vars = [], 0
        for obj in gc.get_objects():
            if isinstance(obj, DDF):
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

    def get_task_opt_type(self, uuid_task):
        return self.tasks_map[uuid_task].get('optimization', self.OPT_OTHER)

    def get_task_function(self, uuid_task):
        return self.tasks_map[uuid_task]['function']

    def set_task_function(self, uuid_task, data):
        self.tasks_map[uuid_task]['function'] = data

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

    def set_auxiliary_variables(self, variables, current_task):
        id_var = self.get_var_by_task(variables, current_task)
        if len(id_var) == 0:
            raise Exception("Variable is already deleted")

        id_var = id_var[0]
        tasks_list = variables[id_var].task_list
        id_task = tasks_list.index(current_task)
        tasks_list = tasks_list[id_task:]

        return id_var, tasks_list

    def is_ready_to_run(self, id_parents):
        return all([self.get_task_status(p) == 'COMPLETED' for p in id_parents])

    def set_operation(self, child_task, id_parents):

        # get the operation to be executed
        task_and_operation = self.get_task_function(child_task)

        # some operations need a schema information
        if self.tasks_map[child_task].get('info', False):
            task_and_operation[1]['info'] = []
            for p in id_parents:
                sc = self.schemas_map[p]
                if isinstance(sc, list):
                    sc = merge_info(sc)
                    sc = compss_wait_on(sc)
                    self.schemas_map[p] = sc
                task_and_operation[1]['info'].append(sc)

        return task_and_operation

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
            print("topological_tasks: ", topological_tasks)

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

                if DEBUG:
                    print(" - task {} ({})".format(
                            self.get_task_name(child_task), child_task[:8]))
                    print("id_parents: {}".format(id_parents))

                # when has parents: wait all parents tasks be completed
                if not self.is_ready_to_run(id_parents):
                    break

                # get input data from parents
                inputs = [self.get_task_function(id_p) for id_p in id_parents]

                # end this path
                if self.get_task_name(child_task) == 'sync':
                    self.run_sync(child_task, id_parents, inputs)
                    break

                elif self.get_task_opt_type(child_task) == self.OPT_OTHER:
                    self.run_opt_others_tasks(child_task, id_parents, inputs)

                elif self.get_task_opt_type(child_task) == self.OPT_SERIAL:
                    self.run_opt_serial_tasks(i_task, child_task, tasks_list,
                                              selected_tasks, inputs)

                elif self.get_task_opt_type(child_task) == self.OPT_LAST:
                    self.run_opt_last_tasks(i_task, child_task, tasks_list,
                                            selected_tasks, inputs, id_parents)

    def run_sync(self, child_task, id_parents, inputs):
        """
        Execute the sync task, in this context, it means that all COMPSs tasks
        until this current sync will be executed.
        This do not represent a COMPSs synchronization.
        """
        if DEBUG:
            print("RUNNING sync ({}) - condition 2.".format(child_task))

        # sync tasks always will have only one parent
        id_p, result = id_parents[0], inputs[0]

        self.set_task_function(child_task, result)
        self.schemas_map[child_task] = self.schemas_map[id_p]

        self.set_task_status(child_task, 'COMPLETED')

    def run_opt_others_tasks(self, child_task, id_parents, inputs):
        """
        The current operation can not be grouped with other operations, so,
        it must be executed separated.
        """

        if DEBUG:
            print("RUNNING {} ({}) - condition 3.".format(
                    self.tasks_map[child_task]['name'], child_task))

        operation = self.set_operation(child_task, id_parents)

        # execute this operation that returns a dictionary
        output_dict = self._execute_task(operation, inputs)

        self.save_opt_others_tasks(output_dict, child_task)

    def run_opt_serial_tasks(self, i_task, child_task, tasks_list,
                             selected_tasks, inputs):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = set(), list()

        if DEBUG:
            print("RUNNING {} ({}) - condition 4.".format(
                    self.tasks_map[child_task]['name'], child_task))

        for id_j, task_opt in enumerate(tasks_list[i_task:]):
            if DEBUG:
                print('Checking optimization type: {} -> {}'.format(
                        task_opt[:8], self.tasks_map[task_opt]['name']))

            group_uuids.add(task_opt)
            group_func.append(self.get_task_function(task_opt))

            if (i_task + id_j + 1) < len(tasks_list):
                next_task = tasks_list[i_task + id_j + 1]

                if selected_tasks.count(task_opt) != \
                        selected_tasks.count(next_task):
                    break

                if not all([self.get_task_opt_type(task_opt) == self.OPT_SERIAL,
                            self.get_task_opt_type(next_task) == self.OPT_SERIAL
                            ]):
                    break

        if DEBUG:
            print("Stages (optimized): {}".format(group_uuids))
            print("opt_functions", group_func)

        file_serial_function = any(['file_in' in self.get_task_name(uid)
                                    for uid in group_uuids])
        result, info = self._execute_serial_tasks(group_func, inputs,
                                                  file_serial_function)
        self.save_lazy_states(result, info, group_uuids)

    def run_opt_last_tasks(self, i_task, child_task, tasks_list,
                           selected_tasks, inputs, id_parents):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = set(), list()

        if DEBUG:
            print("RUNNING {} ({}) - condition 5.".format(
                    self.tasks_map[child_task]['name'], child_task))

        n_input = self.get_n_input(child_task)
        group_uuids.add(child_task)
        operation = self.set_operation(child_task, id_parents)
        group_func.append(operation)

        i_task += 1
        for id_j, task_opt in enumerate(tasks_list[i_task:]):
            if DEBUG:
                print('Checking optimization type: {} -> {}'.format(
                        task_opt[:8], self.tasks_map[task_opt]['name']))

            group_uuids.add(task_opt)
            group_func.append(self.get_task_function(task_opt))

            if (i_task + id_j + 1) < len(tasks_list):
                next_task = tasks_list[i_task + id_j + 1]

                if selected_tasks.count(task_opt) != \
                        selected_tasks.count(next_task):
                    break

                if not all([self.get_task_opt_type(task_opt) == self.OPT_SERIAL,
                            self.get_task_opt_type(next_task) == self.OPT_SERIAL
                            ]):
                    break

        if DEBUG:
            print("Stages (optimized): {}".format(group_uuids))
            print("opt_functions", group_func)

        result, info = self._execute_opt_last_tasks(group_func, inputs, n_input)
        self.save_lazy_states(result, info, group_uuids)

    def save_opt_others_tasks(self, output_dict, child_task):
        # Results in non 'optimization-other' tasks are in dictionary format

        keys_r, keys_i = output_dict['key_data'], output_dict['key_info']

        siblings = self.get_task_sibling(child_task)

        for f, (id_t, key_r, key_i) in enumerate(zip(siblings, keys_r, keys_i)):
            result, info = output_dict[key_r], output_dict[key_i]

            # save results in task_map and schemas_map
            self.schemas_map[id_t] = info
            self.set_task_function(id_t, result)
            self.set_task_status(id_t, 'COMPLETED')

    def save_lazy_states(self, result, info, opt_uuids):

        for o in opt_uuids:
            self.set_task_function(o, result)
            self.schemas_map[o] = info
            self.set_task_status(o, 'COMPLETED')

            if DEBUG:
                print("{} ({}) is COMPLETED - condition 4."
                      .format(self.tasks_map[o]['name'], o[:8]))

    @staticmethod
    def _execute_task(env, input_data):
        """
        Used to execute all non-lazy functions.

        :param env: a list that contains the current task and its parameters.
        :param input_data: A list of DataFrame as input data
        :return:
        """

        function, settings = env

        if len(input_data) == 1:
            input_data = input_data[0]

        output = function(input_data, settings)
        return output

    @staticmethod
    def _execute_serial_tasks(opt, input_data, type_function):

        """
        Used to execute a group of lazy tasks. This method submit
        multiple 'context.task_bundle', one for each data fragment.

        :param opt: sequence of functions and parameters to be executed in
            each fragment
        :param input_data: A list of DataFrame as input data
        :param type_function: if False, use task_bundle otherwise
         task_bundle_file
        :return:
        """

        if len(input_data) == 1:
            input_data = input_data[0]

        nfrag = len(input_data)

        result = [[] for _ in range(nfrag)]
        info = result[:]

        if type_function:
            function = task_bundle_file
        else:
            function = task_bundle

        for f, df in enumerate(input_data):
            result[f], info[f] = function(df, opt, f)

        return result, info

    def _execute_opt_last_tasks(self, opt, data, n_input):

        """
        Used to execute a group of lazy tasks. This method submit
        multiple 'context.task_bundle', one for each data fragment.

        :param opt: sequence of functions and parameters to be executed in
            each fragment
        :param data: input data
        :return:
        """

        fist_task, opt = opt[0], opt[1:]
        tmp1 = None
        if n_input == 1:
            tmp, settings = self._execute_task(fist_task, data)
        else:
            tmp, tmp1, settings = self._execute_task(fist_task, data)
        nfrag = len(tmp)

        opt[0][1] = settings

        result = [[] for _ in range(nfrag)]
        info = result[:]

        if n_input == 1:
            for f, df in enumerate(tmp):
                result[f], info[f] = task_bundle(df, opt, f)
        else:
            for f in range(nfrag):
                result[f], info[f] = task_bundle2(tmp[f], tmp1[f], opt, f)

        return result, info


@task(returns=2)
def task_bundle(data, stage, id_frag):
    return _bundle(data, stage, id_frag)


@task(returns=2, filename=FILE_IN)
def task_bundle_file(data, stage, id_frag):
    return _bundle(data, stage, id_frag)


@task(returns=2)
def task_bundle2(data1, data2, stage, id_frag):
    data = [data1, data2]
    return _bundle(data, stage, id_frag)


def _bundle(data, stage, id_frag):
    info = None
    for f, current_task in enumerate(stage):
        function, settings = current_task
        if isinstance(settings, dict):
            settings['id_frag'] = id_frag
        data, info = function(data, settings)

    return data, info
