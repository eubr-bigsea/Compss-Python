#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""

from ddf_library.bases.metadata import CatalogTask, Status, OPTGroup
from ddf_library.bases.tasks import *
from ddf_library.bases.monitor.monitor import gen_data
from ddf_library.bases.optimizer.ddf_optimizer import DDFOptimizer
from pycompss.api.api import compss_open, compss_wait_on

from ddf_library.utils import merge_info, check_serialization, delete_result,\
    create_stage_files

import copy
import time
import networkx as nx
from prettytable import PrettyTable


class ContextBase(object):

    app_folder = None
    started = False

    catalog_tasks = CatalogTask()

    DEBUG = False
    monitor = False

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

        print("\nRelevant tasks:", t, '\n')

    @staticmethod
    def gen_status():
        n_tasks = n_cached = n_tmp = 0

        for k in ContextBase.catalog_tasks.list_all():
            not_init = ContextBase.catalog_tasks.get_task_name(k) != 'init'
            status = ContextBase.catalog_tasks.get_task_status(k)

            if not_init:
                n_tasks += 1
            if status == Status.STATUS_PERSISTED:
                n_cached += 1
            if status in Status.STATUS_COMPLETED and not_init:
                n_tmp += 1

        table = [['Number of tasks', n_tasks],
                 ['Number of Persisted tasks', n_cached],
                 ['Number of temporary output', n_tmp]]

        return table

    @staticmethod
    def plot_graph():
        from ddf_library.bases.monitor.monitor import select_colors

        for k, _ in ContextBase.catalog_tasks.dag.nodes(data=True):
            status = ContextBase.catalog_tasks.get_task_status(k)
            name = ContextBase.catalog_tasks.get_task_name(k)

            ContextBase.catalog_tasks.dag.nodes[k]['style'] = 'filled'
            ContextBase.catalog_tasks.dag.nodes[k]['label'] = name

            color = select_colors(status)

            if name == 'init':
                color = 'black'
                ContextBase.catalog_tasks.dag.nodes[k]['style'] = 'solid'

            ContextBase.catalog_tasks.dag.nodes[k]['color'] = color

        from networkx.drawing.nx_agraph import write_dot
        t = time.localtime()
        write_dot(ContextBase.catalog_tasks.dag,
                  'DAG_{}.dot'.format(time.strftime('%b-%d-%Y_%H%M', t)))

    @staticmethod
    def start_monitor():
        from ddf_library.bases.monitor.monitor import Monitor
        ContextBase.monitor = Monitor()

    def set_operation(self, child_task, id_parents):

        # get the operation to be executed
        task_and_operation = self.catalog_tasks.get_task_function(child_task)

        # some operations need a schema information
        if self.catalog_tasks.get_info_condition(child_task):
            task_and_operation[1]['info'] = []
            for p in id_parents:
                sc = self.catalog_tasks.get_schema(p)
                if isinstance(sc, list):
                    sc = merge_info(sc)
                    sc = compss_wait_on(sc)
                    self.catalog_tasks.set_schema(p, sc)
                task_and_operation[1]['info'].append(sc)

        return task_and_operation

    def check_pointer(self, last_node):
        """
        Generate a new lineage excluding the computed sub-flows.

        :param last_node: The last task that we want to compute.
        :return: a sorted list starting from previous results (if exists);
        """

        lineage = list(nx.algorithms.topological_sort(self.catalog_tasks.dag))
        idx = lineage.index(last_node)
        lineage = lineage[0:idx+1]

        # check tasks with status different from STATUS_WAIT or STATUS_DELETED
        check_computed = [t for t in lineage
                          if self.catalog_tasks.get_task_status(t) not in
                          [Status.STATUS_WAIT, Status.STATUS_DELETED]]

        # remove those tasks from the temporary dag. we need remove those nodes
        # by graph analyses because if we consider only the topological order
        # we could remove non computed nodes (e.g., in case of join).
        dag = copy.deepcopy(self.catalog_tasks.dag)
        if len(check_computed) > 0:
            for n in check_computed:
                dag.remove_node(n)

        # when we cut the dag (previous code), we might generate multiple
        # subgraphs. We are interested in a connected components where the
        # last node is inside (that, by definition, will exist only one).
        cc = list(nx.connected_components(dag.to_undirected()))
        sub_graph = [c for c in cc if last_node in c][0]

        # create the final lineage based on sub_graph
        reduced_list = [t for t in lineage if t in sub_graph]
        return reduced_list

    @staticmethod
    def update_status(lineage, new_status):

        for uuid in lineage:
            current_status = ContextBase.catalog_tasks.get_task_status(uuid)

            if new_status == Status.STATUS_WAIT:
                if current_status == Status.STATUS_DELETED:
                    ContextBase.catalog_tasks.set_task_status(uuid, new_status)

            elif new_status == Status.STATUS_PERSISTED:
                if current_status == Status.STATUS_COMPLETED:
                    ContextBase.catalog_tasks.set_task_status(uuid, new_status)
                    ContextBase.catalog_tasks.rm_completed(uuid)

            elif new_status == Status.STATUS_COMPLETED:
                if current_status == Status.STATUS_PERSISTED:
                    ContextBase.catalog_tasks.set_task_status(uuid, new_status)
                    ContextBase.catalog_tasks.add_completed(uuid)

    def run_workflow(self, last_task):
        """
        Find the flow of non executed tasks. This method is executed only when
        an action (show, save, cache, etc) is submitted. DDF have lazy
        evaluation system which means that tasks are not executed until a
        result is required (by an action task).

        :param last_task: the current DDF state to be executed
        """

        lineage = self.check_pointer(last_task)

        # optimizer = DDFOptimizer(lineage, self.dag,
        #                          self.catalog_tasks,
        #                          self.catalog_tasks.schema)
        # optimizer.interpreter()
        # optimizer.explain_lineage()

        # all DELETED task are now waiting a new result
        self.update_status(lineage, Status.STATUS_WAIT)

        if ContextBase.DEBUG:
            self.show_workflow(self.catalog_tasks, lineage)

        jump = 0
        # iterate over all sorted tasks
        for i_task, current_task in enumerate(lineage):

            if jump == 0:

                id_parents = self.catalog_tasks.get_task_parents(current_task)
                inputs = self.catalog_tasks.get_input_data(id_parents)
                opt_type = self.catalog_tasks.get_task_opt_type(current_task)

                if ContextBase.DEBUG:
                    msg = "[CONTEXT] Task {} ({}) with parents {}\n" \
                          "[CONTEXT] RUNNING {} as {}"\
                            .format(self.catalog_tasks.
                                    get_task_name(current_task),
                                    current_task[:8], id_parents,
                                    self.catalog_tasks.
                                    get_task_name(current_task), opt_type)
                    print(msg)

                if opt_type == OPTGroup.OPT_OTHER:
                    self.run_opt_others(current_task, id_parents, inputs)

                elif opt_type == OPTGroup.OPT_SERIAL:
                    jump = self.run_opt_serial(lineage[i_task:], inputs)

                elif opt_type == OPTGroup.OPT_LAST:
                    jump = self.run_opt_last(current_task, lineage[i_task:],
                                             inputs, id_parents)

                current_task = lineage[i_task + jump]

            elif jump > 0:
                jump -= 1

            self.delete_computed_results(current_task)

            if self.monitor:
                table = ContextBase.gen_status()
                title = ContextBase.app_folder.replace('/tmp/ddf_', '')
                gen_data(ContextBase.catalog_tasks, table, title)

    def is_removable(self, id_task):
        """
        We keep all non computed tasks or tasks that have a non computed
        children. In other words, in order to delete a task, its degree
        need be more than zero and its children need to be already computed.

        :param id_task:
        :return:
        """

        cond = False
        # take care to not delete data from leaf nodes
        degree = -1 if id_task not in self.catalog_tasks.dag.nodes \
            else self.catalog_tasks.dag.out_degree(id_task)

        if degree > 0:
            # we need to take care operations with siblings
            siblings = self.catalog_tasks.get_task_sibling(id_task)
            cond = True
            for current_sibling in siblings:
                out_edges = self.catalog_tasks.dag.out_edges(current_sibling)
                for (inv, outv) in out_edges:
                    if self.catalog_tasks.get_task_status(outv) == \
                            Status.STATUS_WAIT:
                        return False

        return cond

    def delete_computed_results(self, current_task):
        """
        Delete results from computed tasks in order to free up disk space.

        :param current_task:
        :return:
        """

        for id_task in self.catalog_tasks.list_completed():

            if id_task != current_task and self.is_removable(id_task):

                if ContextBase.DEBUG:
                    print(" - delete_computed_results - {} ({})"
                          .format(self.catalog_tasks.get_task_name(id_task),
                                  id_task[:8]))

                data = self.catalog_tasks.get_task_return(id_task)
                if check_serialization(data):
                    delete_result(data)

                self.catalog_tasks.set_task_status(id_task,
                                                   Status.STATUS_DELETED)
                self.catalog_tasks.set_task_result(id_task, None)
                self.catalog_tasks.rm_schema(id_task)
                ContextBase.catalog_tasks.rm_completed(id_task)

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
                    .format(self.catalog_tasks.get_task_name(task_opt),
                            task_opt[:8])
                print(msg)

            group_uuids.append(task_opt)
            group_func.append(self.catalog_tasks.get_task_function(task_opt))

            if (id_j + 1) < len(lineage):
                next_task = lineage[id_j + 1]

                if not all([self.catalog_tasks.get_task_opt_type(task_opt)
                            == OPTGroup.OPT_SERIAL,
                            self.catalog_tasks.get_task_opt_type(next_task)
                            == OPTGroup.OPT_SERIAL
                            ]):
                    break

                if not (self.catalog_tasks.dag.out_degree(task_opt) ==
                        self.catalog_tasks.dag.in_degree(next_task) == 1):
                    break

                if task_opt not in self.catalog_tasks.\
                        get_task_parents(next_task):
                    break

        if ContextBase.DEBUG:
            names = [self.catalog_tasks.get_task_name(i) for i in group_uuids]
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

        n_input = self.catalog_tasks.get_n_input(child_task)
        group_uuids.append(child_task)
        operation = self.set_operation(child_task, id_parents)
        group_func.append(operation)

        lineage = lineage[1:]
        for id_j, task_opt in enumerate(lineage):
            if ContextBase.DEBUG:
                print(' - Checking optimization type for {} ({})'.format(
                        self.catalog_tasks.get_task_name(task_opt),
                        task_opt[:8]))

            group_uuids.append(task_opt)
            group_func.append(self.catalog_tasks.get_task_function(task_opt))

            if (id_j + 1) < len(lineage):
                next_task = lineage[id_j + 1]

                if not(self.catalog_tasks.dag.out_degree(task_opt) ==
                       self.catalog_tasks.dag.in_degree(next_task) == 1):
                    break

                if not all([self.catalog_tasks.get_task_opt_type(task_opt)
                            == OPTGroup.OPT_SERIAL,
                            self.catalog_tasks.get_task_opt_type(next_task)
                            == OPTGroup.OPT_SERIAL
                            ]):
                    break

                if task_opt not in self.catalog_tasks.\
                        get_task_parents(next_task):
                    break

        if ContextBase.DEBUG:
            names = [self.catalog_tasks.get_task_name(i) for i in group_uuids]
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
        siblings = self.catalog_tasks.get_task_sibling(child_task)

        for f, (id_t, key_r, key_i) in enumerate(zip(siblings, keys_r, keys_i)):
            result, info = output_dict[key_r], output_dict[key_i]

            # save results in task_map and catalog_schemas
            self.catalog_tasks.set_schema(id_t, info)
            self.catalog_tasks.set_task_result(id_t, result)
            self.catalog_tasks.set_task_status(id_t, Status.STATUS_COMPLETED)

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
            self.catalog_tasks.set_task_status(o, Status.STATUS_COMPLETED)

        if 'save' not in self.catalog_tasks.get_task_name(last_uuid):
            self.catalog_tasks.set_task_result(last_uuid, result)
            self.catalog_tasks.set_schema(last_uuid, info)
            self.catalog_tasks.set_task_status(last_uuid,
                                               Status.STATUS_COMPLETED)

        else:
            self.catalog_tasks.set_task_status(last_uuid,
                                               Status.STATUS_PERSISTED)

    @staticmethod
    def _execute_other_task(env, input_data):
        """
        Execute all tasks that cannot be grouped.

        :param env: a list that contains the current task and its parameters.
        :param input_data: A list of DataFrame as input data
        :return:
        """
        try:
            function, settings = env
        except TypeError as e:
            print(e)
            raise Exception('[CONTEXT] Task was not computed or was deleted.')

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

        first_task_name = self.catalog_tasks.get_task_name(uuid_list[0])
        last_task_name = self.catalog_tasks.get_task_name(uuid_list[-1])

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

        last_task_name = self.catalog_tasks.get_task_name(uuid_list[-1])

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
            .ddf_add_task('init', opt=OPTGroup.OPT_OTHER,
                          n_input=0, status=Status.STATUS_COMPLETED,
                          parent=[], function=None)
        return first_uuid

    @staticmethod
    def ddf_add_task(name, opt, function, parent, n_input=-1, n_output=1,
                     status='WAIT', result=None, info=False,
                     info_data=None, expr=None):
        """
        Insert a DDF task in COMPSs Context catalog.

        :param name: DDF task name;
        :param opt: Optimization police;
        :param function: a tuple (function, parameters);
        :param parent: uuid parent task;
        :param n_input: number of parents;
        :param n_output: number of results that this task generates;
        :param status: current status;
        :param result: a list of files output, if completed;
        :param info: True if this task needs information about parents to run;
        :param info_data: Information (schema), if task is already completed;
        :param expr: Information used by DDF Optimizer;
        :return:
        """

        # Add information about the operation if operation is not cataloged yet
        args_task = {
            'optimization': opt,
            'output': n_output,
            'input': n_input if n_input >= 0 else len(parent),
            'info': info,
        }
        ContextBase.catalog_tasks.add_definition(name, args_task)

        if expr is None:
            expr = {}

        if 'name' not in expr:
            expr['name'] = name  # Todo: change it in future

        args_task = {
            'name': name,
            'status': status,
            'function': function,
            'result': result,
            'expr': expr  # TODO: remove or establish it
        }
        new_state_uuid = ContextBase.catalog_tasks.gen_new_uuid()

        ContextBase.catalog_tasks.set_new_task(new_state_uuid, args_task)

        if info_data:
            ContextBase.catalog_tasks.set_schema(new_state_uuid, info_data)

        if status == Status.STATUS_COMPLETED and name != 'init':
            ContextBase.catalog_tasks.add_completed(new_state_uuid)

        # linking parents
        ContextBase.catalog_tasks.dag.add_node(new_state_uuid)
        for p in parent:
            ContextBase.catalog_tasks.dag.add_edge(p, new_state_uuid)

        return new_state_uuid

    @staticmethod
    def link_siblings(siblings):
        """
        Link two or more tasks as siblings, meaning, the result for both task
        are generated together.

        :param siblings: uuid list
        :return:
        """
        for uuid in siblings:
            ContextBase.catalog_tasks.set_task_sibling(uuid, siblings)
