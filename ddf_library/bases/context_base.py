#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""

from ddf_library.bases.tasks import *
from pycompss.api.api import compss_open, compss_wait_on

from ddf_library.utils import merge_info, check_serialization, delete_result,\
    create_stage_files, _gen_uuid

import copy
import time
import networkx as nx
from prettytable import PrettyTable


class ContextBase(object):

    app_folder = None
    started = False
    catalog_schemas = dict()
    catalog_tasks = dict()
    dag = nx.DiGraph()

    OPT_SERIAL = 'serial'  # it can be grouped with others operations
    OPT_OTHER = 'other'  # it can not be performed any kind of task optimization
    OPT_LAST = 'last'  # it contains two or more stages,
    # but only the last stage can be grouped

    STATUS_WAIT = 'WAIT'
    STATUS_DELETED = 'DELETED'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_PERSISTED = 'PERSISTED'  # to persist in order to reuse later

    DEBUG = False

    """
    task_map: a dictionary to stores all following information about a task:

     - name: task name;
     - status: WAIT, COMPLETED, PERSISTED
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
    def gen_status():
        n_tasks = n_cached = n_tmp = 0

        for k in ContextBase.catalog_tasks:
            not_init = ContextBase.catalog_tasks[k]['name'] != 'init'
            status = ContextBase.catalog_tasks[k]['status']
            if not_init:
                n_tasks += 1
            if status == 'PERSISTED':
                n_cached += 1
            if status in 'COMPLETED' and not_init:
                n_tmp += 1

        t = PrettyTable(['Metric', 'Value'])
        t.add_row(['Number of tasks', n_tasks])
        t.add_row(['Number of Persisted tasks', n_cached])
        t.add_row(['Number of temporary output', n_tmp])
        print("\nContext status:\n", t, '\n')

    @staticmethod
    def plot_graph():

        for k, _ in ContextBase.dag.nodes(data=True):
            status = ContextBase.catalog_tasks[k].get('status',
                                                      ContextBase.STATUS_WAIT)
            ContextBase.dag.nodes[k]['style'] = 'filled'
            if ContextBase.dag.nodes[k]['label'] == 'init':
                color = 'black'
                ContextBase.dag.nodes[k]['style'] = 'solid'
            elif status == ContextBase.STATUS_DELETED:
                color = 'lightgray'
            elif status == ContextBase.STATUS_WAIT:
                color = 'yellow'
            elif status == ContextBase.STATUS_PERSISTED:
                color = 'forestgreen'
            else:  # completed
                color = 'lightblue'

            ContextBase.dag.nodes[k]['color'] = color

        from networkx.drawing.nx_agraph import write_dot
        t = time.localtime()
        write_dot(ContextBase.dag,
                  'DAG_{}.dot'.format(time.strftime('%b-%d-%Y_%H%M', t)))

    def create_dag(self, specials=None):
        """
        Create a DAG
        :return: networkx directed Graph
        """

        if specials is None:
            specials = [k for k in self.catalog_tasks]

        for t in self.catalog_tasks:
            parents = self.get_task_parents(t)
            if t in specials:
                self.dag.add_node(t, label=self.get_task_name(t))
                for p in parents:
                    if p in specials:
                        self.dag.add_node(p, label=self.get_task_name(p))
                        self.dag.add_edge(p, t)

    def check_action(self, uuid_task):
        return self.catalog_tasks[uuid_task]['name'] in ['save', 'sync']

    def get_task_name(self, uuid_task):
        return self.catalog_tasks[uuid_task]['name']

    def get_task_opt_type(self, uuid_task):
        return self.catalog_tasks[uuid_task].get('optimization', self.OPT_OTHER)

    def get_task_function(self, uuid_task):
        return self.catalog_tasks[uuid_task]['function']

    def get_task_return(self, uuid_task):
        return self.catalog_tasks[uuid_task].get('result', [])

    def set_task_function(self, uuid_task, data):
        self.catalog_tasks[uuid_task]['function'] = data

    def set_task_result(self, uuid_task, data):
        self.catalog_tasks[uuid_task]['result'] = data

    def get_task_status(self, uuid_task):
        return self.catalog_tasks[uuid_task]['status']

    def set_task_status(self, uuid_task, status):
        self.catalog_tasks[uuid_task]['status'] = status

    def get_task_parents(self, uuid_task):
        return self.catalog_tasks[uuid_task].get('parent', [])

    def get_n_input(self, uuid_task):
        return self.catalog_tasks[uuid_task]['input']

    def get_task_sibling(self, uuid_task):
        return self.catalog_tasks[uuid_task].get('sibling', [uuid_task])

    def get_input_data(self, id_parents):
        return [self.get_task_return(id_p) for id_p in id_parents]

    def set_operation(self, child_task, id_parents):

        # get the operation to be executed
        task_and_operation = self.get_task_function(child_task)

        # some operations need a schema information
        if self.catalog_tasks[child_task].get('info', False):
            task_and_operation[1]['info'] = []
            for p in id_parents:
                sc = self.catalog_schemas[p]
                if isinstance(sc, list):
                    sc = merge_info(sc)
                    sc = compss_wait_on(sc)
                    self.catalog_schemas[p] = sc
                task_and_operation[1]['info'].append(sc)

        return task_and_operation

    def check_pointer(self, lineage):
        """
        Generate a new lineage excluding the computed sub-flows.

        :param lineage: a sorted list of tasks;
        :return: a sorted list starting from previous results (if exists);
        """
        # check tasks with status different from WAIT
        check_computed = [t for t in lineage
                          if self.get_task_status(t) not in
                          [self.STATUS_WAIT, self.STATUS_DELETED]]

        # remove those tasks from the current dag
        dag = copy.deepcopy(self.dag)
        if len(check_computed) > 0:
            for n in check_computed:
                dag.remove_node(n)

        # retrieve all connected components where the
        # last task (action) is inside
        last_node = lineage[-1]
        cc = list(nx.connected_components(dag.to_undirected()))
        sub_graph = [c for c in cc if last_node in c][0]

        reduced_list = [t for t in lineage if t in sub_graph]
        return reduced_list

    def run_workflow(self, lineage):
        """
        Find the flow of non executed tasks. This method is executed only when
        an action (show, save, cache, etc) is submitted. DDF have lazy
        evaluation system which means that tasks are not executed until a
        result is required (by an action task).

        :param lineage: the current DDF state to be executed
        """

        self.create_dag(lineage)
        lineage = self.check_pointer(lineage)

        if ContextBase.DEBUG:
            self.show_workflow(self.catalog_tasks, lineage)

        jump = 0
        # iterate over all sorted tasks
        for i_task, current_task in enumerate(lineage):

            if jump == 0:
                id_parents = self.get_task_parents(current_task)
                inputs = self.get_input_data(id_parents)
                opt_type = self.get_task_opt_type(current_task)

                if ContextBase.DEBUG:
                    msg = "[CONTEXT] Task {} ({}) with parents {}\n" \
                          "[CONTEXT] RUNNING {} as {}"\
                            .format(self.get_task_name(current_task),
                                    current_task[:8], id_parents,
                                    self.get_task_name(current_task), opt_type)
                    print(msg)

                if opt_type == self.OPT_OTHER:
                    self.run_opt_others(current_task, id_parents, inputs)

                elif opt_type == self.OPT_SERIAL:
                    jump = self.run_opt_serial(lineage[i_task:], inputs)

                elif opt_type == self.OPT_LAST:
                    jump = self.run_opt_last(current_task, lineage[i_task:],
                                             inputs, id_parents)

                current_task = lineage[i_task + jump]

            elif jump > 0:
                jump -= 1

            self.delete_computed_results(current_task, lineage)

    def is_removable(self, id_task, current_task):
        """
        We keep all non computed tasks or tasks that have a non computed
        children. In other words, in order to delete a task, its degree
        need be more than zero and its children need to be already computed.

        :param id_task:
        :param current_task:
        :return:
        """

        cond = False
        if self.get_task_status(id_task) == self.STATUS_COMPLETED:
            # take care to not delete data from leaf nodes
            degree = -1 if id_task not in self.dag.nodes \
                else self.dag.out_degree(id_task)

            if degree > 0:
                siblings = self.get_task_sibling(current_task)
                has_siblings = len(siblings) > 1
                # to not delete a split-operation #TODO
                if not has_siblings:
                    out_edges = self.dag.out_edges(id_task)
                    cond = True
                    for (inv, outv) in out_edges:
                        if self.get_task_status(outv) == self.STATUS_WAIT:
                            return False
                        elif 'save' in self.get_task_name(outv) \
                                and len(out_edges) == 1:
                            return False
        return cond

    def delete_computed_results(self, current_task, lineage):
        """
        Delete results from computed tasks in order to free up disk space.

        :param current_task:
        :param lineage:
        :return:
        """

        for id_task in lineage:

            if id_task == current_task:
                return 1

            elif self.is_removable(id_task, current_task):

                if ContextBase.DEBUG:
                    print(" - delete_computed_results - {} ({})"
                          .format(self.get_task_name(id_task), id_task[:8]))

                data = self.get_task_return(id_task)
                if check_serialization(data):
                    delete_result(data)

                self.set_task_status(id_task, self.STATUS_DELETED)
                self.set_task_result(id_task, None)
                self.catalog_schemas.pop(id_task, None)

    def run_opt_others(self, child_task, id_parents, inputs):
        """
        Run operations that currently can not be grouped with other operations,
        so, it must be executed separated.
        """

        operation = self.set_operation(child_task, id_parents)
        # execute this operation that returns a dictionary
        output_dict = self._execute_other_task(operation, inputs)
        self.save_opt_others_tasks(output_dict, child_task)

    def run_opt_serial(self, lineage, inputs):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        In order to group two tasks in a stage: both need to be 'serial'; and
        can not have a branch in the flow between them.
        """
        group_uuids, group_func = list(), list()

        for id_j, task_opt in enumerate(lineage):
            if ContextBase.DEBUG:
                msg = ' - Checking optimization type for {} ({})'\
                    .format(self.get_task_name(task_opt), task_opt[:8])
                print(msg)

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

        if ContextBase.DEBUG:
            names = [self.get_task_name(i) for i in group_uuids]
            ids = [i[0:8] for i in group_uuids]
            msg = " - Stages (optimized): {}" \
                  " - opt_functions: {}".format(ids, names)
            print(msg)

        result, info = self._execute_serial_tasks(group_uuids, group_func,
                                                  inputs)
        self.save_serial_states(result, info, group_uuids)
        jump = len(group_func)-1
        return jump

    def run_opt_last(self, child_task, lineage, inputs, id_parents):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = list(), list()

        n_input = self.get_n_input(child_task)
        group_uuids.append(child_task)
        operation = self.set_operation(child_task, id_parents)
        group_func.append(operation)

        lineage = lineage[1:]
        for id_j, task_opt in enumerate(lineage):
            if ContextBase.DEBUG:
                print(' - Checking optimization type for {} ({})'.format(
                        self.catalog_tasks[task_opt]['name'],
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

        if ContextBase.DEBUG:
            names = [self.get_task_name(i) for i in group_uuids]
            ids = [i[0:8] for i in group_uuids]
            msg = " - Stages (optimized): {}" \
                  " - opt_functions: {}".format(ids, names)
            print(msg)

        result, info = self._execute_opt_last_tasks(group_func,
                                                    group_uuids.copy(),
                                                    inputs, n_input)
        self.save_serial_states(result, info, group_uuids)
        jump = len(group_func)-1
        return jump

    def save_opt_others_tasks(self, output_dict, child_task):
        # Results in non 'optimization-other' tasks are in dictionary format

        # get the keys where data and info are stored in the dictionary
        keys_r, keys_i = output_dict['key_data'], output_dict['key_info']

        # Get siblings uuids. Some operations (currently, only 'split'
        # operation is supported) have more than one output (called siblings).
        # In this case, those operations create a branch starting of them.
        # However, internally, both outputs must be generated together
        # (to performance), and because of that, when a task generates siblings,
        # they are update together.
        siblings = self.get_task_sibling(child_task)

        for f, (id_t, key_r, key_i) in enumerate(zip(siblings, keys_r, keys_i)):
            result, info = output_dict[key_r], output_dict[key_i]

            # save results in task_map and catalog_schemas
            self.catalog_schemas[id_t] = info
            self.set_task_result(id_t, result)
            self.set_task_status(id_t, self.STATUS_COMPLETED)

    def save_serial_states(self, result, info, opt_uuids):
        """
        All tasks in stage must be updated in order to remove obsolete tasks
        in the future.

        :param result:
        :param info:
        :param opt_uuids:
        :return:
        """
        last_uuid = opt_uuids[-1]

        for o in opt_uuids:
            self.set_task_status(o, self.STATUS_DELETED)

        if 'save' not in self.get_task_name(last_uuid):
            self.set_task_result(last_uuid, result)
            self.catalog_schemas[last_uuid] = info
            self.set_task_status(last_uuid, self.STATUS_COMPLETED)
        else:
            self.set_task_status(last_uuid, self.STATUS_PERSISTED)

    @staticmethod
    def _execute_other_task(env, input_data):
        """
        Execute all tasks that cannot be grouped.

        :param env: a list that contains the current task and its parameters.
        :param input_data: A list of DataFrame as input data
        :return:
        """
        function, settings = env
        nfrag = len(input_data)

        if nfrag == 1:
            input_data = input_data[0]

        if ContextBase.DEBUG:
            msg = ' - running task by _execute_other_task\n' \
                  '   * input file {}'.format(input_data)
            print(msg)

        output = function(input_data, settings)
        return output

    def _execute_serial_tasks(self, uuid_list, tasks_list, input_data):

        """
        Execute a group of 'serial' tasks. This method submit
        multiple COMPSs tasks `stage_*in_*out`, one for each data fragment.

        :param uuid_list: sequence of tasks uuid to be executed
        :param tasks_list: sequence of functions and parameters to be executed
         in each fragment
        :param input_data: A list of DataFrame as input data
        :return:
        """

        if len(input_data) == 1:
            input_data = input_data[0]
        nfrag = len(input_data)
        info = [[] for _ in range(nfrag)]

        first_task_name = self.get_task_name(uuid_list[0])
        last_task_name = self.get_task_name(uuid_list[-1])

        if 'read-many-file' == first_task_name:
            if 'save-hdfs' == last_task_name:
                function = stage_1in_0out
            else:
                function = stage_1in_1out

        elif 'read-hdfs' == first_task_name:
            if 'save-hdfs' == last_task_name:
                function = stage_0in_0out
            else:
                function = stage_0in_1out

        elif 'save-hdfs' == last_task_name:
            function = stage_1in_0out
        else:
            function = stage_1in_1out

        if 'save' in last_task_name:
            out_files = tasks_list[-1][1]['output'](nfrag)
            tasks_list[-1][1]['output'] = out_files
        else:
            out_files = create_stage_files(nfrag)

        if ContextBase.DEBUG:
            msg = ' - running task by _execute_serial_tasks\n' \
                  '   * input file {}\n' \
                  '   * output file {}'.format(input_data, out_files)
            print(msg)

        for f, (in_file, out_file) in enumerate(zip(input_data, out_files)):
            info[f] = function(in_file, tasks_list, f, out_file)

        if 'save-file' == last_task_name:
            # Currently, we need to `compss_open` each output file generated by
            # `save-file` operation in order to COMPSs retrieve this output
            # in master node before the end of the `runcompss`.
            for out_file in out_files:
                compss_open(out_file).close()

        return out_files, info

    def _execute_opt_last_tasks(self, tasks_list, uuid_list, data, n_input):

        """
        Execute a group of tasks starting by a 'last' tasks. Some operations
        have more than one processing stage (e.g., sort), some of them, can be
        organized in two stages: the first is called `last` and  `second` is
        called `serial`. In this case, the operations in `last` means that can
        not be grouped with the current flow of tasks (ending the current
        stage), but them, starting from `serial` part, will start a new stage.

        :param tasks_list: sequence of functions and parameters to be executed
         in each fragment
        :param uuid_list: sequence of tasks uuid to be executed
        :param data: input data
        :return:
        """

        first_task, tasks_list = tasks_list[0], tasks_list[1:]
        out_tmp2 = None
        if n_input == 1:
            out_tmp, settings = self._execute_other_task(first_task, data)
        else:
            out_tmp, out_tmp2, settings = \
                self._execute_other_task(first_task, data)
            if out_tmp2 is None:
                # some operations, like geo_within does not return 2 data
                n_input = 1

        # some `last` stages do not change the input data, so we cannot delete
        # them at end.
        intermediate_result = settings.get('intermediate_result', True)
        nfrag = len(out_tmp)

        # after executed the first stage, we update the next task settings
        tasks_list[0][1] = settings
        info = [[] for _ in range(nfrag)]

        last_task_name = self.get_task_name(uuid_list[-1])

        if 'save' in last_task_name:
            out_files = tasks_list[-1][1]['output'](nfrag)
            tasks_list[-1][1]['output'] = out_files
        else:
            out_files = create_stage_files(nfrag)

        if n_input == 1:

            if 'save-hdfs' == last_task_name:
                function = stage_1in_0out
            else:
                function = stage_1in_1out

            for f, (in_file, out_file) in enumerate(zip(out_tmp, out_files)):
                info[f] = function(in_file, tasks_list, f, out_file)

        else:
            if 'save-hdfs' == last_task_name:
                function = stage_2in_0out
            else:
                function = stage_2in_1out

            for f, (in_file1, in_file2, out_file) in \
                    enumerate(zip(out_tmp, out_tmp2, out_files)):
                info[f] = function(in_file1, in_file2, tasks_list, f, out_file)
            # removing temporary tasks
            if intermediate_result:
                delete_result(out_tmp2)

        # removing temporary tasks
        if intermediate_result:
            delete_result(out_tmp)

        return out_files, info

    @staticmethod
    def create_init():
        first_uuid = ContextBase\
            .ddf_add_task('init', opt=ContextBase.OPT_OTHER,
                          n_input=0, status=ContextBase.STATUS_COMPLETED,
                          parent=[], function=None)
        return first_uuid

    @staticmethod
    def ddf_add_task(name, opt, function, parent, n_input=-1, n_output=1,
                     status='WAIT', result=None, info=False,
                     info_data=None):
        # TODO: doc this
        if n_input < 0:
            n_input = len(parent)

        args_task = {
            'name': name,
            'status': status,
            'optimization': opt,
            'function': function,
            'parent': parent,
            'output': n_output,
            'input': n_input,
            'info': info,
        }

        new_state_uuid = _gen_uuid()
        while new_state_uuid in ContextBase.catalog_tasks:
            new_state_uuid = _gen_uuid()

        ContextBase.catalog_tasks[new_state_uuid] = args_task
        if info_data:
            ContextBase.catalog_schemas[new_state_uuid] = info_data

        if result:
            ContextBase.catalog_tasks[new_state_uuid]['result'] = result

        return new_state_uuid

    @staticmethod
    def link_siblings(siblings):
        #TODO: DOC this
        for uuid in siblings:
            if uuid not in ContextBase.catalog_tasks:
                raise Exception('uuid "" not in catalog_tasks'.format(uuid[:8]))
            ContextBase.catalog_tasks[uuid]['sibling'] = siblings
