#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd
import numpy as np


class SortOperation(object):

    def transform(self, data, settings):
        """
        Returns a DataFrame sorted by the specified column(s).

        :param data: A list with nfrag pandas's dataframe;
        :param settings: A dictionary that contains:
            - columns: The list of columns to be sorted.
            - ascending: A list indicating whether the sort order
                is ascending (True) for each column.
            - algorithm: 'batcher' to Batcher odd–even mergesort
                (default if nfrag is power of 2), 'oddeven' to
                commom Odd-Even (if nfrag is not a power of 2);
        :return: A list with nfrag pandas's dataframe

        """

        nfrag = len(data)
        settings = self.preprocessing(settings)
        algorithm = settings.get('algorithm', 'batcher')

        def is_power2(num):
            """states if a number is a power of two."""
            return num != 0 and ((num & (num - 1)) == 0)

        if nfrag == 1:
            info = [[] for _ in range(nfrag)]
            result = [[] for _ in range(nfrag)]
            for f in range(nfrag):
                result[f], info[f] = partial_sort(data[f], settings)

        elif is_power2(nfrag) and algorithm is 'batcher':
            result, info = self._sort_by_batcher(data, settings)
        else:
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

    def _sort_by_batcher(self, data, settings):
        """
        Batcher's odd–even mergesort is a generic construction devised by Ken
        Batcher for sorting networks of size O(n (log n)2) and depth
        O((log n)2), where n is the number of items to be sorted.
        """
        data, info = self.batcher_oddeven_mergesort(data, settings)
        return data, info

    def batcher_oddeven_mergesort(self, data, settings):
        """
        The odd-even mergesort algorithm was developed by K.E. Batcher [
        Bat 68]. It is based on a merge algorithm that merges two sorted
        halves of a sequence to a completely sorted sequence.
        """
        nfrag = len(data)
        info = [[] for _ in range(nfrag)]
        pairs_to_compare = self.oddeven_merge_sort_range(0, nfrag - 1)

        for i, j in pairs_to_compare:
            data[i], data[j], info[i], info[j] = \
                _mergesort_bit(data[i], data[j], settings)

        return data, info

    def oddeven_merge(self, lo, hi, r):
        step = r * 2
        if step < hi - lo:
            for i in self.oddeven_merge(lo, hi, step):
                yield i
            for i in self.oddeven_merge(lo + r, hi, step):
                yield i
            for i in [(i, i + r) for i in range(lo + r, hi - r, step)]:
                yield i
        else:
            yield (lo, lo + r)

    def oddeven_merge_sort_range(self, lo, hi):
        if (hi - lo) >= 1:
            # if there is more than one element, split the input
            # down the middle and first sort the first and second
            # half, followed by merging them.
            mid = lo + ((hi - lo) // 2)
            for i in self.oddeven_merge_sort_range(lo, mid):
                yield i
            for i in self.oddeven_merge_sort_range(mid + 1, hi):
                yield i
            for i in self.oddeven_merge(lo, hi, 1):
                yield i


@task(returns=2)
def partial_sort(data, settings):
    cols = settings['columns']
    order = settings['ascending']

    data.sort_values(cols, inplace=True, ascending=order, kind='mergesort')
    data.reset_index(drop=True, inplace=True)

    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]

    return data, info


@task(returns=4)
def _mergesort_bit(data1, data2, settings):
    """Return 1 if [data1, data2] is sorted, otherwise is -1.

    Space complexity: 2(N+M)
    """
    cols = settings['columns']
    order = settings['ascending']

    n1 = len(data1)
    n2 = len(data2)

    data = pd.concat([data1, data2], sort=False, ignore_index=True)
    data.sort_values(cols, inplace=True, ascending=order, kind='mergesort')

    data1 = data.head(n1)
    data1.reset_index(drop=True, inplace=True)

    data2 = data.tail(n2)
    data2.reset_index(drop=True, inplace=True)

    info1 = [data1.columns.tolist(), data1.dtypes.values, [len(data1)]]
    info2 = [data2.columns.tolist(), data2.dtypes.values, [len(data2)]]

    return data1, data2, info1, info2


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









