#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.bases.context_base import ContextBase
from ddf_library.utils import merge_schema
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
    STATUS_DELETED = 'DELETED'

    def __init__(self):
        self.last_uuid = 'not_defined'

    def _ddf_initial_setup(self, data, info=False):
        tmp = data.cache()
        data.last_uuid = tmp.last_uuid
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
            if ContextBase.catalog_tasks[self.last_uuid]['status'] not in \
                    [ContextBase.STATUS_WAIT, ContextBase.STATUS_DELETED]:
                self.partitions = \
                    ContextBase.catalog_tasks[self.last_uuid]['result']
                stored = True
                break
            else:
                ContextBase().run_workflow(self.last_uuid)

        if not stored:
            raise Exception("[ERROR] - _check_stored - cache cant be done")
