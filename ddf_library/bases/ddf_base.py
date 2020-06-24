#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.bases.metadata import Status
from ddf_library.bases.context_base import ContextBase


class DDFSketch(object):

    """
    Basic functions that are necessary when submit a new operation
    """

    def __init__(self):
        self.last_uuid = 'not_defined'

    def _ddf_initial_setup(self, data, info=False):
        tmp = data.cache()
        data.last_uuid = tmp.last_uuid
        df = ContextBase.catalog_tasks.get_task_return(data.last_uuid).copy()
        nfrag = len(df)
        if info:
            info = self._get_info()
            return df, nfrag, data, info
        else:
            return df, nfrag, data

    def _get_info(self):

        self._check_stored()
        info = ContextBase.catalog_tasks.get_merged_schema(self.last_uuid)
        return info

    def _check_stored(self):
        """

        :return: Check if ddf variable is currently executed.
        """
        stored = False
        for _ in range(2):
            if ContextBase.catalog_tasks.get_task_status(self.last_uuid) \
                    not in [Status.STATUS_WAIT, Status.STATUS_DELETED]:
                self.partitions = ContextBase.\
                    catalog_tasks.get_task_return(self.last_uuid)
                stored = True
                break
            else:
                self.last_uuid = ContextBase().run_workflow(self.last_uuid)

        if not stored:
            raise Exception("[ERROR] - _check_stored - cache cant be done")
