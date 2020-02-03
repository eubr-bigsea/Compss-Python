#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""

from ddf_library.bases.context_base import CONTEXTBASE
from ddf_library.utils import check_serialization, \
    create_stage_files, save_stage_file, read_stage_file, delete_result

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT

import copy
import networkx as nx

DEBUG = False


class COMPSsContext(CONTEXTBASE):
    """
    Controls the DDF tasks executions
    """
    def stop(self):
        """To avoid that COMPSs sends back all partial result at end."""
        for id_task in list(self.tasks_map.keys()):
            data = self.get_task_return(id_task)

            if check_serialization(data):
                delete_result(data)

            self.tasks_map.pop(id_task)
            self.catalog.pop(id_task, None)

        COMPSsContext.catalog = dict()
        COMPSsContext.tasks_map = dict()
        COMPSsContext.dag = nx.DiGraph()

    @staticmethod
    def set_log(enabled=True):
        global DEBUG
        DEBUG = enabled

    def check_pointer(self, lineage):
        """
        Generate a new lineage excluding the computed sub-flows.

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
                inputs = self.get_input_data(id_parents)
                opt_type = self.get_task_opt_type(current_task)

                if DEBUG:
                    msg = "[CONTEXT] Task {} ({}) with parents {}\n"\
                            .format(self.get_task_name(current_task),
                                    current_task[:8], id_parents)
                    msg += "[CONTEXT] RUNNING {} as {}"\
                        .format(self.get_task_name(current_task), opt_type)
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

            self.delete_old_tasks(current_task, lineage)

    def is_removable(self, id_task, current_task):

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
        return cond

    def delete_old_tasks(self, current_task, lineage):
        """
        We keep all tasks that is not computed yet or that have a not computed
        children.
        :param current_task:
        :param lineage:
        :return:

        o degree tem que ser maior que 0, mas tbm tem q saber se o filho já foi
        computado para casos como join (em que uma tarefa pode esperar )
        """

        for id_task in lineage:

            if id_task == current_task:
                return 1
            elif self.is_removable(id_task, current_task):

                if DEBUG:
                    print(" - delete_old_tasks - {} ({})"
                          .format(self.tasks_map[id_task]['name'],
                                  id_task[:8]))

                data = self.get_task_return(id_task)
                if check_serialization(data):
                    delete_result(data)

                self.set_task_status(id_task, self.STATUS_WAIT)
                self.set_task_result(id_task, None)
                self.catalog.pop(id_task, None)

    def run_opt_others(self, child_task, id_parents, inputs):
        """
        The current operation can not be grouped with other operations, so,
        it must be executed separated.
        """

        operation = self.set_operation(child_task, id_parents)
        # execute this operation that returns a dictionary
        output_dict = self._execute_task(operation, inputs)

        if self.get_task_name(child_task) == 'save':
            self.catalog[child_task] = \
                self.catalog.get(id_parents[0], None)
            self.set_task_result(child_task, inputs[0])
            self.set_task_status(child_task, self.STATUS_COMPLETED)

        else:
            self.save_opt_others_tasks(output_dict, child_task)

    def run_opt_serial(self, lineage, inputs):
        """
        The current operation can be grouped with other operations. This method
        check if the next operations share this behavior. If it does, group
        them to execute together, otherwise, execute it as a single task.
        """
        group_uuids, group_func = list(), list()

        for id_j, task_opt in enumerate(lineage):
            if DEBUG:
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

        if DEBUG:
            names = [self.get_task_name(i) for i in group_uuids]
            ids = [i[0:8] for i in group_uuids]
            msg = " - Stages (optimized): {}" \
                  " - opt_functions: {}".format(ids, names)
            print(msg)

        file_serial_function = 0
        if any(['load_text-file_in' in self.get_task_name(uid)
                for uid in group_uuids]):
            file_serial_function = 1
        elif any(['load_text-0in' in self.get_task_name(uid)
                  for uid in group_uuids]):
            file_serial_function = 2

        result, info = self._execute_serial_tasks(group_func, inputs,
                                                  file_serial_function)
        self.save_lazy_states(result, info, group_uuids)
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
            names = [self.get_task_name(i) for i in group_uuids]
            ids = [i[0:8] for i in group_uuids]
            msg = " - Stages (optimized): {}" \
                  " - opt_functions: {}".format(ids, names)
            print(msg)

        result, info = self._execute_opt_last_tasks(group_func, inputs,
                                                    n_input)
        self.save_lazy_states(result, info, group_uuids)
        jump = len(group_func)-1
        return jump

    def save_opt_others_tasks(self, output_dict, child_task):
        # Results in non 'optimization-other' tasks are in dictionary format

        keys_r, keys_i = output_dict['key_data'], output_dict['key_info']

        siblings = self.get_task_sibling(child_task)    #TODO: ENTENDER

        for f, (id_t, key_r, key_i) in enumerate(zip(siblings, keys_r, keys_i)):
            result, info = output_dict[key_r], output_dict[key_i]

            # save results in task_map and catalog
            self.catalog[id_t] = info
            self.set_task_result(id_t, result)
            self.set_task_status(id_t, self.STATUS_COMPLETED)

    def save_lazy_states(self, result, info, opt_uuids):
        """
        All list must be updated in order to remove obsolete tasks in the future

        :param result:
        :param info:
        :param opt_uuids:
        :return:
        """
        last_uuid = opt_uuids[-1]
        self.set_task_result(last_uuid, result)
        self.catalog[last_uuid] = info

        for o in opt_uuids:
            self.set_task_status(o, self.STATUS_COMPLETED)

    @staticmethod
    def _execute_task(env, input_data):
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

        if DEBUG:
            msg = ' - running task by _execute_task\n' \
                  '   * input file {}'.format(input_data)
            print(msg)

        output = function(input_data, settings)
        return output

    @staticmethod
    def _execute_serial_tasks(tasks_list, input_data, type_function):

        """
        Used to execute a group of lazy tasks. This method submit
        multiple 'context.task_bundle', one for each data fragment.

        :param tasks_list: sequence of functions and parameters to be executed in
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
        out_files = create_stage_files(nfrag)

        if type_function == 0:
            function = task_bundle_1parquet_1parquet
        elif type_function == 1:
            function = task_bundle_1csv_1parquet
        elif type_function == 2:
            function = task_bundle_0in_1parquet
        else:
            raise Exception("[CONTEXT] - ERROR in _execute_serial_tasks")

        if DEBUG:
            msg = ' - running task by _execute_serial_tasks\n' \
                  '   * input file {}\n' \
                  '   * output file {}'.format(input_data,out_files)
            print(msg)

        for f, (in_file, out_file) in enumerate(zip(input_data, out_files)):
            info[f] = function(in_file, tasks_list, f, out_file)

        return out_files, info

    def _execute_opt_last_tasks(self, opt, data, n_input):

        """
        Used to execute a group of lazy tasks. This method submit
        multiple 'context.task_bundle', one for each data fragment.

        :param opt: sequence of functions and parameters to be executed in
            each fragment
        :param data: input data
        :return:
        """

        first_task, opt = opt[0], opt[1:]
        out_tmp2 = None
        if n_input == 1:
            out_tmp, settings = self._execute_task(first_task, data)
        else:
            out_tmp, out_tmp2, settings = self._execute_task(first_task, data)
            if out_tmp2 is None:
                # some operations, like geo_within does not return 2 data
                n_input = 1

        intermediate_result = settings.get('intermediate_result', False)
        nfrag = len(out_tmp)

        opt[0][1] = settings

        info = [[] for _ in range(nfrag)]
        out_files = create_stage_files(nfrag)

        if n_input == 1:
            for f, (in_file, out_file) in enumerate(zip(out_tmp, out_files)):
                info[f] = task_bundle_1parquet_1parquet(in_file, opt, f,
                                                        out_file)

        else:
            for f, (in_file1, in_file2, out_file) in \
                    enumerate(zip(out_tmp, out_tmp2, out_files)):
                info[f] = task_bundle_2parquet_1parquet(in_file1, in_file2,
                                                        opt, f, out_file)
            # removing temporary tasks
            if intermediate_result:
                delete_result(out_tmp2)

        # removing temporary tasks
        if intermediate_result:
            delete_result(out_tmp)

        return out_files, info


@task(input_file=FILE_IN, output_file=FILE_OUT, returns=1)
def task_bundle_1parquet_1parquet(input_file, stage, id_frag, output_file):
    """
    Will perform most functions with the serial tag. Task has 1 data input
    and return 1 data output with its schema

    :param input_file: Input filepath;
    :param stage: a list with functions and its parameters;
    :param id_frag: Block index
    :param output_file: Output filepath;
    :return:
    """

    # by using parquet, we can specify each column we want to read
    columns = None
    if stage[0][0].__name__ == 'task_select':
        columns = stage[0][1]['columns']

    data = read_stage_file(input_file, columns)
    data, info = _bundle(data, stage, id_frag)
    save_stage_file(output_file, data)
    return info


@task(input_file1=FILE_IN, input_file2=FILE_IN, output_file=FILE_OUT, returns=1)
def task_bundle_2parquet_1parquet(input_file1, input_file2, stage,
                                  id_frag, output_file):
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


@task(returns=1, data_input=FILE_IN, data_output=FILE_OUT)   # TODO: Test!
def task_bundle_1csv_1parquet(data_input, stage, id_frag, data_output):
    """
    Used to read a folder from common file system.

    :param data_input:
    :param stage:
    :param id_frag:
    :param data_output:
    :return:
    """

    data, info = _bundle(data_input, stage, id_frag)
    save_stage_file(data_output, data)
    return info


@task(returns=1,  data_output=FILE_OUT)   # TODO: Test!
def task_bundle_0in_1parquet(data, stage, id_frag, data_output):
    """
    Used to read files from HDFS.

    :param data:
    :param stage:
    :param id_frag:
    :param data_output:
    :return:
    """
    data, info = _bundle(data, stage, id_frag)
    save_stage_file(data_output, data)
    return info


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
