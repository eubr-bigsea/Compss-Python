#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd
import numpy as np

#TODO: bittonic sort


class SortOperation(object):

    def transform(self, data, settings):
        """
        Returns a DataFrame sorted by the specified column(s).

        :param data: A list with nfrag pandas's dataframe;
        :param settings: A dictionary that contains:
            - columns: The list of columns to be sorted.
            - ascending: A list indicating whether the sort order
                is ascending (True) for each column.

        :return: A list with nfrag pandas's dataframe

        ..note: #TODO
        """

        nfrag = len(data)
        settings = self.preprocessing(settings)
        result, info = self._sort_by_oddeven(data, settings, nfrag)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output

    @staticmethod
    def preprocessing(settings):
        """Check all settings."""
        columns = settings.get('columns', [])
        if not isinstance(columns, list):
            columns = [columns]

        if len(columns) == 0:
            raise Exception('`columns` do not must be empty.')

        asc = settings.get('ascending', [])
        if not isinstance(asc, list):
            asc = [asc]

        n1 = len(columns)
        n2 = len(asc)
        if n1 > n2:
            asc = asc + [True for _ in range(n2-n1)]
        elif n2 > n1:
            asc = asc[:n1]

        settings['columns'] = columns
        settings['asceding'] = asc
        return settings

    @staticmethod
    def _sort_by_oddeven(data, settings, nfrag):
        """Sort by Odd-Even Sort."""

        if isinstance(data[0], pd.DataFrame):
            import copy
            result = copy.deepcopy(data)
        else:
            # when using deepcopy and variable is FutureObject
            # list, COMPSs is not able to restore in worker
            result = data[:]

        from pycompss.api.api import compss_wait_on
        info = [[] for _ in range(nfrag)]

        list_sorted = False
        while not list_sorted:
            signals = [[0] for _ in range(nfrag)]

            for i in range(0, nfrag-1, 2):  # odd - even
                result[i], result[i+1], info[i], info[i+1], signals[i] = \
                    _mergesort(result[i], result[i+1], settings)

            for i in range(1, nfrag-1, 2):  # even - odd
                result[i], result[i+1], info[i], info[i+1], signals[i]\
                    = _mergesort(result[i], result[i+1], settings)

            signals = compss_wait_on(signals)
            list_sorted = not any([s[0] == -1 for s in signals])

        return result, info


@task(returns=5)
def _mergesort(data1, data2, settings):
    """Return 1 if [data1, data2] is sorted, otherwise is -1.

    Space complexity: 2(N+M)
    """
    cols = settings['columns']
    order = settings['ascending']+[True]
    cols_all = data1.columns
    n1 = len(data1)
    n2 = len(data2)

    signal = [1]
    if n1 == 0 or n2 == 0:
        info1 = [data1.columns.tolist(), data1.dtypes.values, [len(data1)]]
        info2 = [data2.columns.tolist(), data2.dtypes.values, [len(data2)]]
        return data1, data2, info1, info2, signal

    data = pd.concat([data1, data2], sort=False)
    tmp_col = check_column(cols_all)
    # To check if has any change
    ids = [i for i in range(n1+n2)]
    data[tmp_col] = ids
    cols.append(tmp_col)
    data.sort_values(cols, inplace=True, ascending=order, kind='mergesort')
    ids_sort = data[tmp_col].values.tolist()

    if ids[:n1] != ids_sort[:n1]:
        signal = [-1]
        data1 = data.head(n1)
        data1.drop(columns=[tmp_col], inplace=True)
        data1.reset_index(drop=True, inplace=True)

        data = data.tail(n2)

        data.reset_index(drop=True, inplace=True)
        data2.loc[:] = data.loc[:, cols_all].values
    else:
        if ids[n1:] != ids_sort[n1:]:
            signal = [-1]
            data = data.tail(n2)
            data.reset_index(drop=True, inplace=True)
            data2.loc[:] = data.loc[:, cols_all].values

    info1 = [data1.columns.tolist(), data1.dtypes.values, [len(data1)]]
    info2 = [data2.columns.tolist(), data2.dtypes.values, [len(data2)]]
    return data1, data2, info1, info2, signal


def check_column(cols):
    base = 'tmp-sort'
    i = 0
    col = '{}_{}'.format(base, i)
    while col in cols:
        col = '{}_{}'.format(base, i)
    return col
