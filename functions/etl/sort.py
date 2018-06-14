#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
import pandas as pd
import numpy as np

class SortOperation(object):
    """Sort Operation.

    Returns a DataFrame sorted by the specified column(s).
    """

    def transform(self, data, settings, nfrag):
        """transform.

        :param data: A list with nfrag pandas's dataframe;
        :param settings: A dictionary that contains:
            - columns: The list of columns to be sorted.
            - ascending: A list indicating whether the sort order
                is ascending (True) for each column.
        :param nfrag: The number of fragments;
        :return: A list with nfrag pandas's dataframe

        Note: the list of columns should have the same size of the list
        of boolean to indicating if it is ascending sorting.
        """
        settings = self.preprocessing(settings)
        result = self._sort_by_oddeven(data, settings, nfrag)

        return result

    def preprocessing(self, settings):
        """Check all settings."""
        cols1 = settings.get('columns', [])
        asc = settings.get('ascending', [])
        if any([len(cols1) == 0,
                len(asc) == 0,
                len(cols1) != len(asc)]):
            raise Exception('The list of `columns` ans `ascending` '
                            'should have equal length (and != 0).')

        return settings

    def _sort_by_oddeven(self, data, settings, nfrag):
        """Sort by Odd-Even Sort."""

        if isinstance(data[0], pd.DataFrame):
            import copy
            result = copy.deepcopy(data)
        else:
            # when using deepcopy and variable is FutureObject
            # list, COMPSs is not able to restore in worker
            result = data[:]

        from pycompss.api.api import compss_wait_on

        list_sorted = False
        while not list_sorted:
            signals = [[0] for _ in range(nfrag)]

            for i in range(0, nfrag-1, 2):  # odd - even
                result[i] = _mergesort(result[i], result[i+1],
                                       signals[i], settings)

            for i in range(1, nfrag-1, 2):  # even - odd
                result[i] = _mergesort(result[i], result[i+1],
                                       signals[i], settings)

            signals = compss_wait_on(signals)
            # print signals
            list_sorted = not any([s[0] == -1 for s in signals])

        return result


# @task(signals=INOUT, returns=list)
# def _mergesort(data1, data2, signals, settings):
#     """Return 1 if [data1, data2] is sorted, otherwise is -1."""
#     cols = settings['columns']
#     order = settings['ascending']
#     n1 = len(data1)
#     n2 = len(data2)
#     signals[0] = 1
#
#     if n1 == 0 or n2 == 0:
#         return signals, data1, data2
#
#     data = pd.concat([data1, data2], ignore_index=True)
#
#     tmp = data1[cols].values.flatten()
#     data = data.sort_values(cols, ascending=order)
#
#     data1 = data[:n1]
#
#     if any(data1[cols].values.flatten() != tmp):
#         signals[0] = -1
#
#     print data.index
#
#     idx1 = data.index[:n1]
#     idx2 = data.index[n1:]
#     idx = [idx1,  idx2]
#     print idx
#     return idx
#
#
# @task(returns=list)
# def _apply_sort(data1, data2, idx, f):
#     """Return 1 if [data1, data2] is sorted, otherwise is -1."""
#     data = pd.concat([data1, data2], ignore_index=True)
#     data = data.loc[idx[f]]
#     return data


@task(data2=INOUT, signals=INOUT, returns=list)
def _mergesort(data1, data2, signals, settings):
    """Return 1 if [data1, data2] is sorted, otherwise is -1.

    Space complexity: 2(N+M)
    """
    cols = settings['columns']
    order = settings['ascending']+[True]
    cols_all = data1.columns
    n1 = len(data1)
    n2 = len(data2)

    if n1 == 0 or n2 == 0:
        signals[0] = 1
        return data1

    data = pd.concat([data1, data2], sort=False)
    tmp_col = check_column(cols_all)
    # To check if has any change
    ids = [i for i in range(n1+n2)]
    data[tmp_col] = ids
    cols.append(tmp_col)
    data.sort_values(cols, inplace=True, ascending=order, kind='mergesort')
    ids_sort = data[tmp_col].values.tolist()

    if ids[:n1] != ids_sort[:n1]:
        signals[0] = -1
        data1 = data.head(n1)
        data1.drop(columns=[tmp_col], inplace=True)
        data1.reset_index(drop=True, inplace=True)

        data = data.tail(n2)

        data.reset_index(drop=True, inplace=True)
        data2.loc[:] = data.loc[:, cols_all].values
    else:
        if ids[n1:] != ids_sort[n1:]:
            signals[0] = -1
            data = data.tail(n2)
            data.reset_index(drop=True, inplace=True)
            data2.loc[:] = data.loc[:, cols_all].values
    return data1


def check_column(cols):
    base = 'tmp-sort'
    i = 0
    col = '{}_{}'.format(base, i)
    while col in cols:
        col = '{}_{}'.format(base, i)
    return col
