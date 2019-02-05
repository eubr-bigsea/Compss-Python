#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd


class UnionOperation(object):
    """Union Operation.

    Function which do a union between two pandas dataframes.
    """

    def transform(self, data1, data2):
        """UnionOperation.

        :param data1:   A list with nfrag pandas's dataframe;
        :param data2:   Other list with nfrag pandas's dataframe;
        :param nfrag: The number of fragments;
        :return:        Returns a list with nfrag pandas's dataframe.
        """

        nfrag = len(data1) + len(data2)
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _union(data1[f], data2[f])
        return result

    def transform_serial(self, list1, list2):
        """Perform a partil union."""
        return _union_(list1, list2)


@task(returns=list)
def _union(list1, list2):
    """Perform a partil union."""
    return _union_(list1, list2)


def _union_(list1, list2):
    """Perform a partil union."""
    if len(list1) == 0:
        return list2
    elif len(list2) == 0:
        return list1
    else:
        return pd.concat([list1, list2], ignore_index=True)
