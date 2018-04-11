#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *


class SelectOperation(object):
    """Select Operation.

    Function which do a Projection with the columns choosed.
    """

    def transform(self, data, columns, numFrag):
        """SelectOperation.

        :param data:    A list with numFrag pandas's dataframe;
        :param columns: A list with the columns names which will be selected;
        :param numFrag: A number of fragments;
        :return:        A list with numFrag pandas's dataframe
                        with only the columns choosed.
        """
        result = [[] for f in range(numFrag)]
        if len(columns) > 0:
            for f in range(numFrag):
                result[f] = self._select(data[f], columns)
        else:
            raise Exception("You should pass at least one column.")

        return result

    @task(isModifier=False, returns=list)
    def _select(self, list1, fields):
        """Perform a partial projection."""
        # remove the columns that not in list1
        fields = [field for field in fields if field in list1.columns]
        if len(fields) == 0:
            raise Exception("The columns passed as parameters "
                            "do not belong to this dataframe.")
        return list1[fields]

    def select_serial(self, list1, fields):
        """Perform a partial projection."""
        # remove the columns that not in list1
        fields = [field for field in fields if field in list1.columns]
        if len(fields) == 0:
            raise Exception("The columns passed as parameters "
                            "do not belong to this dataframe.")
        return list1[fields]
