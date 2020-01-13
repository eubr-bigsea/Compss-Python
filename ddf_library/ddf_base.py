#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.context import COMPSsContext
from ddf_library.utils import merge_schema, _gen_uuid

from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on


class DDFSketch(object):

    """
    Basic functions that are necessary when submit a new operation
    """
    OPT_SERIAL = 'serial'  # it can be grouped with others operations
    OPT_OTHER = 'other'  # it can not be performed any kind of task optimization
    OPT_LAST = 'last'  # it contains two or more stages,
    # but only the last stage can be grouped

    optimization_ops = [OPT_OTHER, OPT_SERIAL, OPT_LAST]

    def __init__(self):

        self.last_uuid = 'not_defined'
        self.settings = dict()
        pass

    @staticmethod
    def _merge_tasks_list(seq):
        """
        Merge two list of tasks removing duplicated tasks

        :param seq: list with possible duplicated elements
        :return:
        """
        seen = set()
        return [x for x in seq if x not in seen and not seen.add(x)]

    @staticmethod
    def _generate_uuid():
        """
        Generate a unique id
        :return: uuid
        """
        new_state_uuid = _gen_uuid()
        while new_state_uuid in COMPSsContext.tasks_map:
            new_state_uuid = _gen_uuid()
        return new_state_uuid

    @staticmethod
    def _ddf_initial_setup(data):
        tmp = data.cache()
        data.task_list, data.last_uuid = tmp.task_list, tmp.last_uuid
        df = COMPSsContext.tasks_map[data.last_uuid]['result'].copy()
        nfrag = len(df)
        return df, nfrag, data

    def _get_info(self):

        self._check_stored()
        info = COMPSsContext.catalog[self.last_uuid]
        if isinstance(info, list):
            if not isinstance(info[0], list):
                info = merge_reduce(merge_schema, info)
        info = compss_wait_on(info)

        COMPSsContext.catalog[self.last_uuid] = info
        return info

    def _check_stored(self):
        """

        :return: Check if ddf variable is currently executed.
        """
        stored = False

        for _ in range(2):
            if COMPSsContext.tasks_map[self.last_uuid]['status'] in \
                [COMPSsContext.STATUS_COMPLETED,
                 COMPSsContext.STATUS_PERSISTED,
                 COMPSsContext.STATUS_MATERIALIZED,
                 COMPSsContext.STATUS_TEMP_VIEW]:
                self.partitions = \
                    COMPSsContext.tasks_map[self.last_uuid]['result']
                stored = True
                break
            else:
                self._run_compss_context()

        if not stored:
            raise Exception("[ERROR] - _check_stored - cache cant be done")

    def _ddf_add_task(self, task_name, opt, function, parent, n_output=1,
                      n_input=1, status='WAIT', info=None, result=None):

        uuid_key = self._generate_uuid()
        COMPSsContext.tasks_map[uuid_key] = {
            'name': task_name,
            'status': status,
            'optimization': opt,
            'function': function,
            'parent': parent,
            'output': n_output,
            'input': n_input
        }

        if info:
            COMPSsContext.catalog[uuid_key] = info
        if result:
            COMPSsContext.tasks_map[uuid_key]['result'] = result
        return uuid_key

    def _run_compss_context(self):
        COMPSsContext().run_workflow(self.task_list)
        return self
