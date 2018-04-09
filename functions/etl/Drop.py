#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Drop.

Returns a new DataFrame that drops the specified column.
Nothing is done if schema doesn't contain the given column name(s).
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *

class DropOperation(object):

    def __init__(self):
        pass

    def transform(self, data, columns, numFrag):
        """transform.

        :param data: A list with numFrag pandas's dataframe;
        :param columns: A list with the columns names to be removed;
        :param numFrag: A number of fragments;
        :return: A list with numFrag pandas's dataframe.
        """
        result = [[] for f in range(numFrag)]
        for f in range(numFrag):
            result[f] = self._drop(data[f], columns)

        return result

    @task(returns=list)
    def _drop(self, df, columns):
        """Peform a partial drop operation."""
        return df.drop(columns, axis=1)


    def drop_serial(self, df, columns):
        """Peform a partial drop operation."""
        return df.drop(columns, axis=1)
