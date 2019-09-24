#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""

from ddf_library.utils import merge_info, check_serialization

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN
from pycompss.api.api import compss_wait_on, compss_delete_object

import copy
import networkx as nx

DEBUG = False


class COMPSsContext(object):
    """
    Controls the DDF tasks executions
    """
    adj_tasks = dict()
    schemas_map = dict()
    tasks_map = dict()
    dag = nx.DiGraph()

    OPT_SERIAL = 'serial'  # it can be grouped with others operations
    OPT_OTHER = 'other'  # it can not be performed any kind of task optimization
    OPT_LAST = 'last'  # it contains two or more stages,
    # but only the last stage can be grouped

    STATUS_WAIT = 'WAIT'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_TEMP_VIEW = 'TEMP_VIEWED'  # temporary

    STATUS_PERSISTED = 'PERSISTED'
    STATUS_MATERIALIZED = 'MATERIALIZED'  # persisted

    optimization_ops = [OPT_OTHER, OPT_SERIAL, OPT_LAST]
    """
    task_map: a dictionary to stores all following information about a task:

     - name: task name;
     - status: WAIT or COMPLETED;
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

    def set_log(self, enabled=True):
        global DEBUG
        DEBUG = enabled

    def context_status(self):
        n_tasks = sum([1 for k in self.tasks_map
                       if self.tasks_map[k]['name'] != 'init'])
        n_cached = sum([1 for k in self.tasks_map
                        if
                        self.tasks_map[k]['status'] == 'PERSISTED' and
                        self.tasks_map[k]['name'] != 'init'])
        n_materialized = sum([1 for k in self.tasks_map
                              if self.tasks_map[k]['status'] ==
                              'MATERIALIZED' and self.tasks_map[k][
                                  'name']
                              != 'init'])
        n_output = sum([1 for k in self.tasks_map
                        if self.tasks_map[k].get("result", False) and
                        self.tasks_map[k]['name'] != 'init'])
        n_tmp = sum([1 for k in self.tasks_map
                     if self.tasks_map[k]['status']
                     in ['TEMP_VIEWED', 'COMPLETED'] and
                     self.tasks_map[k]['name'] != 'init'])
        print("""
        Number of tasks: {}
        Number of Persisted tasks: {}
        Number of Materialized tasks: {}
        Number of temporary results saved (Temporary view and completed): {}
        Number of output: {}
        """.format(n_tasks, n_cached, n_materialized, n_tmp, n_output))

        self.plot_graph(COMPSsContext.tasks_map, COMPSsContext.dag)

    def show_workflow(self, selected_tasks):
        """
        Show the final workflow. Only to debug
        :param selected_tasks: list of tasks to be executed in this flow.
        """
        print("Relevant tasks:")
        for uuid in selected_tasks:
            print("\t{} - ({}) - {}"
                  .format(self.tasks_map[uuid]['name'],
                          uuid[:8],
                          isinstance(self.tasks_map[uuid].get("result", None),
                                     list)))
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

    @staticmethod
    def plot_graph(tasks_map, dag):

        for k, _ in dag.nodes(data=True):
            status = tasks_map[k].get('status', COMPSsContext.STATUS_WAIT)
            dag.nodes[k]['style'] = 'filled'
            if dag.nodes[k]['label'] == 'init':
                color = 'black'
                dag.nodes[k]['style'] = 'solid'
            elif status == COMPSsContext.STATUS_WAIT:
                color = 'lightgray'
            elif status in [COMPSsContext.STATUS_MATERIALIZED,
                            COMPSsContext.STATUS_PERSISTED]:
                color = 'forestgreen'
            else:  # temp viewed or completed
                color = 'lightblue'

            dag.nodes[k]['color'] = color

        from networkx.drawing.nx_agraph import write_dot
        import time
        t = time.localtime()
        write_dot(dag, 'DAG_{}.dot'.format(time.strftime('%b-%d-%Y_%H%M', t)))

    def create_dag(self, specials=None):
        """
        Create a Adjacency task list
        Parent --> sons
        :return:
        """

        if specials is None:
            specials = [k for k in self.tasks_map]

        for t in self.tasks_map:
            parents = self.tasks_map[t]['parent']
            if t in specials:
                for p in parents:
                    if p in specials:
                        if p not in self.adj_tasks:
                            self.adj_tasks[p] = []
                        self.adj_tasks[p].append(t)

        for k in self.adj_tasks:
            # if self.tasks_map[k].get('result', False):

            self.dag.add_node(k, label=self.get_task_name(k))
            self.adj_tasks[k] = list(set(self.adj_tasks[k]))

            for j in self.adj_tasks[k]:
                self.dag.add_node(j, label=self.get_task_name(j))
                self.dag.add_edge(k, j)

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
        data = [self.get_task_return(id_p) for id_p in id_parents]
        return data

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

    def check_pointer(self, lineage):
        """
        Remove already computed sub-flows.

        :param lineage: a sorted list of tasks;
        :return: a sorted list starting from previous results (if exists);
        """
        check_computed = [t for t in lineage
                          if self.get_task_status(t) != self.STATUS_WAIT]

        dag = copy.deepcopy(self.dag)
        if len(check_computed) > 0:
            for n in check_computed:
                dag.remove_node(n)

        last_node = lineage[-1]
        cc = list(nx.connected_components(dag.to_undirected()))
        sub_graph = [c for c in cc if last_node in c][0]

        reduced_list = [t for t in lineage if t in sub_graph]
        return reduced_list

    def run_workflow(self, ddf_var):
        """
        #!TODO
        Find flow of tasks non executed with an action (show, save, cache, etc)
        and execute each flow until a 'sync' is found. Action tasks represents
        operations which its results will be saw by the user.

        :param ddf_var: the DDF to be executed
        """

        lineage = ddf_var.task_list
        # and perform a topological sort to create a DAG
        self.create_dag(lineage)
        # reverse topological to find the last computed parent
        lineage = self.check_pointer(lineage)

        if DEBUG:
            self.show_workflow(lineage)
            print("topological_tasks: ", lineage)

        jump = 0
        # iterate over all sorted tasks
        for i_task, current_task in enumerate(lineage):

            if jump == 0:
                id_parents = self.get_task_parents(current_task)

                if DEBUG:
                    print(" - task {} ({})".format(
                            self.get_task_name(current_task), current_task[:8]))
                    print("id_parents: {}".format(id_parents))

                # get input data from parents
                inputs = self.get_input_data(id_parents)

                if self.get_task_opt_type(current_task) == self.OPT_OTHER:
                    self.run_opt_others_tasks(current_task, id_parents, inputs)

                elif self.get_task_opt_type(current_task) == self.OPT_SERIAL:
                    jump = self.run_opt_serial_tasks(current_task,
                                                     lineage[i_task:], inputs)

                elif self.get_task_opt_type(current_task) == self.OPT_LAST:
                    jump = self.run_opt_last_tasks(current_task,
                                                   lineage[i_task:],
                                                   inputs, id_parents)
                current_task = lineage[i_task + jump]
                self.delete_old_tasks(current_task)

            elif jump > 0:
                jump -= 1

    def delete_old_tasks(self, current_task):
        # the same thing to schema
        for id_task in self.tasks_map:
            # take care to not delete data from leaf nodes
            degree = -1 if id_task not in self.dag.nodes \
                else self.dag.out_degree(id_task)
            siblings = self.get_task_sibling(current_task)
            has_siblings = len(siblings) > 1

            if degree > 0 and id_task != current_task and not has_siblings:
                if self.get_task_status(id_task) in [self.STATUS_COMPLETED,
                                                     self.STATUS_TEMP_VIEW]:
                    if DEBUG:
                        print("[delete_old_tasks] - id: {} - TASK NAME: {} "
                              "- Degree: {} - siblings : {}"
                              .format(id_task, self.tasks_map[id_task]['name'],
                                      degree, siblings))
                    data = self.tasks_map[id_task].get('result', False)
                    if check_serialization(data):
                        compss_delete_object(data)
                    self.set_task_status(id_task, self.STATUS_WAIT)
                    self.tasks_map[id_task]['result'] = None
                    self.schemas_map.pop(id_task, None)

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

    def run_opt_serial_tasks(self, child_task, lineage, inputs):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = list(), list()

        if DEBUG:
            print("RUNNING {} ({}) - condition 4.".format(
                    self.tasks_map[child_task]['name'], child_task))

        for id_j, task_opt in enumerate(lineage):
            if DEBUG:
                print('Checking optimization type: {} -> {}'.format(
                        task_opt[:8], self.tasks_map[task_opt]['name']))

            group_uuids.append(task_opt)
            group_func.append(self.get_task_function(task_opt))

            if (id_j + 1) < len(lineage):
                next_task = lineage[id_j + 1]

                if not all([self.get_task_opt_type(task_opt) == self.OPT_SERIAL,
                            self.get_task_opt_type(next_task) == self.OPT_SERIAL
                            ]):
                    break

                if not (self.dag.out_degree(task_opt) ==
                        self.dag.in_degree(next_task) == 1):
                    break

        if DEBUG:
            print("Stages (optimized): {}".format(group_uuids))
            print("opt_functions", group_func)

        file_serial_function = any(['file_in' in self.get_task_name(uid)
                                    for uid in group_uuids])
        result, info = self._execute_serial_tasks(group_func, inputs,
                                                  file_serial_function)
        self.save_lazy_states(result, info, group_uuids)
        jump = len(group_func)-1
        return jump

    def run_opt_last_tasks(self, child_task, lineage,
                           inputs, id_parents):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = list(), list()

        if DEBUG:
            print("RUNNING {} ({}) - condition 5.".format(
                    self.tasks_map[child_task]['name'], child_task))

        n_input = self.get_n_input(child_task)
        group_uuids.append(child_task)
        operation = self.set_operation(child_task, id_parents)
        group_func.append(operation)

        lineage = lineage[1:]
        for id_j, task_opt in enumerate(lineage):
            if DEBUG:
                print('Checking optimization type: {} -> {}'.format(
                        task_opt[:8], self.tasks_map[task_opt]['name']))

            group_uuids.append(task_opt)
            group_func.append(self.get_task_function(task_opt))

            if (id_j + 1) < len(lineage):
                next_task = lineage[id_j + 1]

                if not(self.dag.out_degree(task_opt) ==
                       self.dag.in_degree(next_task) == 1):
                    break

                if not all([self.get_task_opt_type(task_opt) == self.OPT_SERIAL,
                            self.get_task_opt_type(next_task) == self.OPT_SERIAL
                            ]):
                    break

        if DEBUG:
            print("Stages (optimized): {}".format(group_uuids))
            print("opt_functions in last", group_func)

        result, info = self._execute_opt_last_tasks(group_func, inputs, n_input)
        self.save_lazy_states(result, info, group_uuids)
        jump = len(group_func)-1
        return jump

    def save_opt_others_tasks(self, output_dict, child_task):
        # Results in non 'optimization-other' tasks are in dictionary format

        keys_r, keys_i = output_dict['key_data'], output_dict['key_info']

        siblings = self.get_task_sibling(child_task)

        for f, (id_t, key_r, key_i) in enumerate(zip(siblings, keys_r, keys_i)):
            result, info = output_dict[key_r], output_dict[key_i]

            # save results in task_map and schemas_map
            self.schemas_map[id_t] = info
            self.set_task_result(id_t, result)
            self.set_task_status(id_t, self.STATUS_COMPLETED)

    def save_lazy_states(self, result, info, opt_uuids):

        for o in opt_uuids:
            self.set_task_result(o, result)
            self.schemas_map[o] = info
            self.set_task_status(o, self.STATUS_COMPLETED)

            if DEBUG:
                print("{} ({}) is SUBMITTED - condition 4 or 5."
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
            function = task_bundle_1inf
        else:
            function = task_bundle_1out

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
                result[f], info[f] = task_bundle_1out(df, opt, f)
        else:
            for f in range(nfrag):
                result[f], info[f] = task_bundle_2in(tmp[f], tmp1[f], opt, f)

        return result, info


# task has 1 data input and return 1 data output
@task(returns=2)
def task_bundle_1out(data, stage, id_frag):
    return _bundle(data, stage, id_frag)


# task where the first execution has 2 inputs data
@task(returns=2)
def task_bundle_2in(data1, data2, stage, id_frag):
    data = [data1, data2]
    return _bundle(data, stage, id_frag)


@task(returns=1, filename=FILE_IN)   # TODO: Test!
def task_bundle_1inf(data, stage, id_frag):
    return _bundle(data, stage, id_frag)


def _bundle(data, stage, id_frag):
    info = None
    for f, current_task in enumerate(stage):
        function, parameters = current_task
        if isinstance(parameters, dict):
            parameters['id_frag'] = id_frag
        data, info = function(data, parameters)

    return data, info
