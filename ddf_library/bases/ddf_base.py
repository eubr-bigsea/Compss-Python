#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.bases.context_base import ContextBase
from ddf_library.utils import merge_schema, _gen_uuid
from pycompss.api.api import compss_wait_on


class DDFSketch(object):

    """
    Basic functions that are necessary when submit a new operation
    """
    OPT_SERIAL = 'serial'  # it can be grouped with others operations
    OPT_OTHER = 'other'  # it can not be performed any kind of task optimization
    OPT_LAST = 'last'  # it contains two or more stages,
    # but only the last stage can be grouped

    STATUS_WAIT = 'WAIT'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_PERSISTED = 'PERSISTED'

    optimization_ops = [OPT_OTHER, OPT_SERIAL, OPT_LAST]

    def __init__(self):

        self.last_uuid = 'not_defined'
        self.settings = dict()
        self.task_list = []
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

    def _ddf_initial_setup(self, data, info=False):
        tmp = data.cache()
        data.task_list, data.last_uuid = tmp.task_list, tmp.last_uuid
        df = ContextBase.catalog_tasks[data.last_uuid]['result'].copy()
        nfrag = len(df)
        if info:
            info = self._get_info()
            return df, nfrag, data, info
        else:
            return df, nfrag, data

    def _get_info(self):

        self._check_stored()
        info = ContextBase.catalog_schemas[self.last_uuid]
        if isinstance(info, list):
            if not isinstance(info[0], list):
                info = merge_schema(info)
        info = compss_wait_on(info)

        ContextBase.catalog_schemas[self.last_uuid] = info
        return info

    def _check_stored(self):
        """

        :return: Check if ddf variable is currently executed.
        """
        stored = False
        for _ in range(2):
            if ContextBase.catalog_tasks[self.last_uuid]['status'] != \
                    ContextBase.STATUS_WAIT:
                self.partitions = \
                    ContextBase.catalog_tasks[self.last_uuid]['result']
                stored = True
                break
            else:
                self._run_compss_context()

        if not stored:
            raise Exception("[ERROR] - _check_stored - cache cant be done")

    def _run_compss_context(self):
        ContextBase().run_workflow(self.task_list)
        return self
