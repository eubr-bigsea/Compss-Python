#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""

from ddf_library.utils import merge_info, check_serialization, \
    create_stage_files, save_stage_file, read_stage_file

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT
from pycompss.api.api import compss_wait_on, compss_delete_object

import copy
import networkx as nx
from prettytable import PrettyTable

DEBUG = False


class COMPSsContext(object):
    """
    Controls the DDF tasks executions
    """
    adj_tasks = dict()
    schemas_map = dict()
    tasks_map = dict()
    dag = nx.DiGraph()
    stage_id = 0

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

    @staticmethod
    def set_log(enabled=True):
        global DEBUG
        DEBUG = enabled

    def context_status(self):
        n_tasks = sum([1 for k in self.tasks_map
                       if self.get_task_name(k) != 'init'])
        n_cached = sum([1 for k in self.tasks_map
                        if self.get_task_status(k) == 'PERSISTED' and
                        self.get_task_name(k) != 'init'])
        n_materialized = sum([1 for k in self.tasks_map
                              if self.get_task_status(k) == 'MATERIALIZED'
                              and self.get_task_name(k) != 'init'])
        n_output = sum([1 for k in self.tasks_map
                        if self.tasks_map[k].get("result", False) and
                        self.get_task_name(k) != 'init'])
        n_tmp = sum([1 for k in self.tasks_map
                     if self.get_task_status(k) in ['TEMP_VIEWED', 'COMPLETED']
                     and self.get_task_name(k) != 'init'])

        t = PrettyTable(['Metric', 'Value'])
        t.add_row(['Number of tasks', n_tasks])
        t.add_row(['Number of Persisted tasks', n_cached])
        t.add_row(['Number of Materialized tasks', n_materialized])
        t.add_row(['Number of temporary results saved '
                   '(Temporary view and completed)', n_tmp])
        t.add_row(['Number of output', n_output])

        print(t)

        self.plot_graph(COMPSsContext.tasks_map, COMPSsContext.dag)

    @staticmethod
    def show_workflow(tasks_map, selected_tasks):
        """
        Show the final workflow. Only to debug
        :param tasks_map: Context of all tasks;
        :param selected_tasks: list of tasks to be executed in this flow.
        """

        t = PrettyTable(['Order', 'Task name', 'uuid', 'Result is stored'])
        for i, uuid in enumerate(selected_tasks):
            t.add_row([i,
                       tasks_map[uuid]['name'],
                       uuid[:8],
                       isinstance(tasks_map[uuid].get("result", None), list)
                       ])
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
        return [self.get_task_return(id_p) for id_p in id_parents]

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

    def run_workflow(self, lineage):
        """
        #!TODO
        Find flow of tasks non executed with an action (show, save, cache, etc)
        and execute each flow until a 'sync' is found. Action tasks represents
        operations which its results will be saw by the user.

        :param lineage: the current DDF state to be executed
        """

        self.create_dag(lineage)
        lineage = self.check_pointer(lineage)

        if DEBUG:
            self.show_workflow(self.tasks_map, lineage)

        jump = 0
        # iterate over all sorted tasks
        for i_task, current_task in enumerate(lineage):

            if jump == 0:
                id_parents = self.get_task_parents(current_task)

                if DEBUG:
                    print("[CONTEXT] Task {} ({}) with parents {}".format(
                            self.get_task_name(current_task),
                            current_task[:8],
                            id_parents))

                # get input data from parents
                inputs = self.get_input_data(id_parents)

                if self.get_task_opt_type(current_task) == self.OPT_OTHER:
                    self.run_opt_others_tasks(current_task, id_parents, inputs,
                                              self.stage_id)
                    self.stage_id += 1

                elif self.get_task_opt_type(current_task) == self.OPT_SERIAL:
                    jump = self.run_opt_serial_tasks(current_task,
                                                     lineage[i_task:], inputs,
                                                     self.stage_id)
                    self.stage_id += 1

                elif self.get_task_opt_type(current_task) == self.OPT_LAST:
                    jump = self.run_opt_last_tasks(current_task,
                                                   lineage[i_task:],
                                                   inputs, id_parents,
                                                   self.stage_id)
                    self.stage_id += 2
                current_task = lineage[i_task + jump]
                self.delete_old_tasks(current_task)

            elif jump > 0:
                jump -= 1

    def check_task_childrens(self, task_opt):
        """
        is possible, when join lineages that a task wil have a children
        in the future. So, its important to keep track on that.
        :return:
        """
        out_edges = self.dag.out_edges(task_opt)
        for (inv, outv) in out_edges:
            if self.get_task_status(outv) == self.STATUS_WAIT:
                return True

        return False

    def delete_old_tasks(self, current_task):
        """
        We keep all tasks that is not computed yet or that have a not computed
        children.
        :param current_task:
        :return:
        """
        # the same thing to schema
        for id_task in self.tasks_map:
            # take care to not delete data from leaf nodes
            degree = -1 if id_task not in self.dag.nodes \
                else self.dag.out_degree(id_task)
            siblings = self.get_task_sibling(current_task)
            has_siblings = len(siblings) > 1
            childrens = self.check_task_childrens(id_task)
            if all([degree > 0,  # do not delete leaf tasks
                    id_task != current_task,  # or if the current task
                    not has_siblings,  # or if is a split
                    not childrens  # or if it has a children that needs its data
                    ]):
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

    def run_opt_others_tasks(self, child_task, id_parents, inputs, stage_id):
        """
        The current operation can not be grouped with other operations, so,
        it must be executed separated.
        """

        if DEBUG:
            print("[CONTEXT] RUNNING {} - run_opt_others_tasks".format(
                    self.tasks_map[child_task]['name']))

        operation = self.set_operation(child_task, id_parents)

        # execute this operation that returns a dictionary
        output_dict = self._execute_task(operation, inputs, stage_id)

        self.save_opt_others_tasks(output_dict, child_task)

    def run_opt_serial_tasks(self, child_task, lineage, inputs, stage_id):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = list(), list()

        if DEBUG:
            print("[CONTEXT] RUNNING {} - run_opt_serial_tasks".format(
                    self.tasks_map[child_task]['name']))

        for id_j, task_opt in enumerate(lineage):
            if DEBUG:
                print(' - Checking optimization type for {} ({})'.format(
                        self.tasks_map[task_opt]['name'],
                        task_opt[:8]))

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

                if task_opt not in self.get_task_parents(next_task):
                    break

        if DEBUG:
            print(" - Stages (optimized): {}".format(group_uuids))
            print(" - opt_functions", group_func)

        file_serial_function = any(['file_in' in self.get_task_name(uid)
                                    for uid in group_uuids])
        result, info = self._execute_serial_tasks(group_func, inputs,
                                                  file_serial_function,
                                                  stage_id)
        self.save_lazy_states(result, info, group_uuids)
        jump = len(group_func)-1
        return jump

    def run_opt_last_tasks(self, child_task, lineage,
                           inputs, id_parents, stage_id):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = list(), list()

        if DEBUG:
            print("[CONTEXT] RUNNING {} - run_opt_last_tasks".format(
                    self.tasks_map[child_task]['name']))

        n_input = self.get_n_input(child_task)
        group_uuids.append(child_task)
        operation = self.set_operation(child_task, id_parents)
        group_func.append(operation)

        lineage = lineage[1:]
        for id_j, task_opt in enumerate(lineage):
            if DEBUG:
                print(' - Checking optimization type for {} ({})'.format(
                        self.tasks_map[task_opt]['name'],
                        task_opt[:8]))

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

                if task_opt not in self.get_task_parents(next_task):
                    break

        if DEBUG:
            print(" - Stages (optimized): {}".format(group_uuids))
            print(" - opt_functions", group_func)

        result, info = self._execute_opt_last_tasks(group_func, inputs,
                                                    n_input, stage_id)
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

    @staticmethod
    def _execute_task(env, input_data, stage_id):
        """
        Used to execute all non-lazy functions.

        :param env: a list that contains the current task and its parameters.
        :param input_data: A list of DataFrame as input data
        :return:
        """

        function, settings = env

        nfrag = len(input_data)
        if nfrag == 1:
            input_data = input_data[0]

        settings['stage_id'] = stage_id

        if DEBUG:
            print(' - running task by _execute_serial_tasks')
            print('   * input file {}'.format(input_data))

        output = function(input_data, settings)
        return output

    @staticmethod
    def _execute_serial_tasks(opt, input_data, type_function, stage_id):

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
        info = [[] for _ in range(nfrag)]
        out_files = create_stage_files(stage_id, nfrag)

        if type_function:
            function = task_bundle_1inf
        else:
            function = task_bundle_1out

        if DEBUG:
            print(' - running task by _execute_serial_tasks')
            print('   * input file {}\n   * output file {}'.format(input_data,
                                                                   out_files))
        for f, (in_file, out_file) in enumerate(zip(input_data, out_files)):
            info[f] = function(in_file, opt, f, out_file)

        return out_files, info

    def _execute_opt_last_tasks(self, opt, data, n_input, stage_id):

        """
        Used to execute a group of lazy tasks. This method submit
        multiple 'context.task_bundle', one for each data fragment.

        :param opt: sequence of functions and parameters to be executed in
            each fragment
        :param data: input data
        :return:
        """

        fist_task, opt = opt[0], opt[1:]
        out_tmp2 = None
        if n_input == 1:
            out_tmp, settings = self._execute_task(fist_task, data, stage_id)
        else:
            out_tmp, out_tmp2, settings = self._execute_task(fist_task, data,
                                                             stage_id)
        nfrag = len(out_tmp)

        opt[0][1] = settings

        info = [[] for _ in range(nfrag)]
        out_files = create_stage_files(stage_id, nfrag)

        if n_input == 1:
            for f, (in_file, out_file) in enumerate(zip(out_tmp, out_files)):
                info[f] = task_bundle_1out(in_file, opt, f, out_file)
        else:
            for f, (in_file1, in_file2, out_file) in \
                    enumerate(zip(out_tmp, out_tmp2, out_files)):
                info[f] = task_bundle_2in(in_file1, in_file2, opt, f, out_file)

        return out_files, info


@task(input_file=FILE_IN, output_file=FILE_OUT, returns=1)
def task_bundle_1out(input_file, stage, id_frag, output_file):
    """
    Will perform most functions with the serial tag. Task has 1 data input
    and return 1 data output with its schema

    :param input_file: Input filepath;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :param output_file: Output filepath;
    :return:
    """
    data = read_stage_file(input_file)
    data, info = _bundle(data, stage, id_frag)
    save_stage_file(output_file, data)
    return info


@task(input_file1=FILE_IN, input_file2=FILE_IN, output_file=FILE_OUT, returns=1)
def task_bundle_2in(input_file1, input_file2, stage, id_frag, output_file):
    """
    Executed when the first task has two inputs.

    :param input_file1: Input filepath 1;
    :param input_file2: Input filepath 2;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :param output_file: Output filepath;
    :return:
    """
    data1 = read_stage_file(input_file1)
    data2 = read_stage_file(input_file2)
    data = [data1, data2]
    data, info = _bundle(data, stage, id_frag)
    save_stage_file(output_file, data)
    return info


@task(returns=1, filename=FILE_IN)   # TODO: Test!
def task_bundle_1inf(data, stage, id_frag):
    return _bundle(data, stage, id_frag)


def _bundle(data, stage, id_frag):
    """
    Base method to process each stage.

    :param data: The input data;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :return: An output data and a schema information
    """
    info = None
    for f, current_task in enumerate(stage):
        function, parameters = current_task
        if isinstance(parameters, dict):
            parameters['id_frag'] = id_frag
        data, info = function(data, parameters)

    return data, info
