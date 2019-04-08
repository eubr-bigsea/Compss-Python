#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


import uuid
from ddf_library.context import COMPSsContext
from ddf_library.utils import merge_schema
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on


class DDFSketch(object):

    """
    Basic functions that are necessary when submit a new operation
    """

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
        new_state_uuid = str(uuid.uuid4())
        while new_state_uuid in COMPSsContext.tasks_map:
            new_state_uuid = str(uuid.uuid4())
        return new_state_uuid

    @staticmethod
    def _set_n_input(state_uuid, idx):
        """
        Method to inform the index of the input data

        :param state_uuid: id of the current task
        :param idx: idx of input data
        :return:
        """

        if 'n_input' not in COMPSsContext.tasks_map[state_uuid]:
            COMPSsContext.tasks_map[state_uuid]['n_input'] = []
        COMPSsContext.tasks_map[state_uuid]['n_input'].append(idx)

    @staticmethod
    def _ddf_inital_setup(data):
        tmp = data.cache()
        n_input = COMPSsContext.tasks_map[tmp.last_uuid]['n_input'][0]
        if n_input == -1:
            n_input = 0
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][n_input]
        nfrag = len(df)
        return df, nfrag, tmp

    def _get_info(self):

        self._check_cache()
        n_input = self.settings['input']
        info = COMPSsContext.schemas_map[self.last_uuid][n_input]
        if isinstance(info, list):
            if not isinstance(info[0], list):
                info = merge_reduce(merge_schema, info)
        info = compss_wait_on(info)

        COMPSsContext.schemas_map[self.last_uuid][n_input] = info
        return info

    def _check_cache(self):
        """

        :return: Check if ddf variable is currently executed.
        """

        cached = False

        for _ in range(2):
            if COMPSsContext.tasks_map[self.last_uuid]['status'] == 'COMPLETED':
                n_input = COMPSsContext.tasks_map[self.last_uuid]['n_input'][0]
                self.partitions = \
                    COMPSsContext.tasks_map[self.last_uuid]['function'][n_input]
                cached = True
                break
            else:
                self.cache()

        if not cached:
            raise Exception("ERROR - toPandas - not cached")

    def _ddf_add_task(self, task_name, status, lazy, function,
                      parent, n_output, n_input, info=None):

        uuid_key = self._generate_uuid()
        COMPSsContext.tasks_map[uuid_key] = {
            'name': task_name,
            'status': status,
            'lazy': lazy,
            'function': function,
            'parent': parent,
            'output': n_output,
            'input': n_input
        }

        if info:
            COMPSsContext.schemas_map[uuid_key] = {0: info}
        return uuid_key

    def _run_compss_context(self, wanted=None):
        COMPSsContext().run_workflow(wanted)
        return self
