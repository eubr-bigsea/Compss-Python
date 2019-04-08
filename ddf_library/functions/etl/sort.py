#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, merge_info
from pycompss.api.task import task
# from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_delete_object, compss_wait_on
from pycompss.api.constraint import constraint
import pandas as pd
import numpy as np
import time
import datetime


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

        ..note: This operation doesnt change the schema.
        """

        nfrag = len(data)
        settings = self.preprocessing(settings)
        algorithm = settings.get('algorithm', 'batcher')

        info = settings['info'][0]
        if nfrag == 1:
            result = [[] for _ in range(nfrag)]
            for f in range(nfrag):
                result[f] = partial_sort(data[f], settings)
        elif self.is_power2(nfrag) and algorithm is 'batcher':
            print("Sort Operation using Batcher odd–even mergesort.")
            result, info = self._sort_by_batcher(data, settings)

        else:
            print("Sort Operation using classical Odd–Even sort.")
            result = self._sort_by_oddeven(data, settings, nfrag)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output

    @staticmethod
    def is_power2(num):
        """states if a number is a power of two."""
        return num != 0 and ((num & (num - 1)) == 0)

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

        list_sorted = False
        while not list_sorted:
            signals = [True for _ in range(nfrag)]

            for i in range(0, nfrag-1, 2):  # odd - even
                result[i], result[i+1], signals[i] = \
                    _mergesort(result[i], result[i+1], settings)

            for i in range(1, nfrag-1, 2):  # even - odd
                result[i], result[i+1], signals[i]\
                    = _mergesort(result[i], result[i+1], settings)

            signals = compss_wait_on(signals)
            list_sorted = all(signals)

        return result

    def _sort_by_batcher(self, data, settings):
        """
        Batcher's odd–even mergesort is a generic construction devised by Ken
        Batcher for sorting networks of size O(n (log n)2) and depth
        O((log n)2), where n is the number of items to be sorted.

        This adaptation only works if each partition have the same number of
        elements. To use it in this scenario, the algorithm creates some
        temporary rows. At the end, these rows are removed.
        """

        from ddf_library.functions.etl.balancer import WorkloadBalancer
        settings['forced'] = True
        output = WorkloadBalancer(settings).transform(data)
        data, info = output['data'], output['info']
        info = merge_info(info)
        info = compss_wait_on(info)

        settings['max_partition'] = max(info['size'])
        settings['info'] = info

        data = self.batcher_oddeven_mergesort(data, settings)
        return data, info

    def batcher_oddeven_mergesort(self, data, settings):
        """
        The odd-even mergesort algorithm was developed by K.E. Batcher [
        Bat 68]. It is based on a merge algorithm that merges two sorted
        halves of a sequence to a completely sorted sequence.
        """
        nfrag = len(data)
        pairs_to_compare = list(self.oddeven_merge_sort_range(0, nfrag - 1))

        # create a dictionary with the first and the last touch in each frag.
        f_seen = check_touchs(pairs_to_compare)
        for i, (v, u) in enumerate(pairs_to_compare):
            info = get_touch(f_seen, v, u, i)
            tmp1, tmp2 = _mergesort_bit(data[v], data[u], settings, info)
            compss_delete_object(data[v])
            compss_delete_object(data[u])
            data[v], data[u] = tmp1, tmp2

        return data

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


def check_touchs(pairs_to_compare):
    f_seen = {}
    for i, (v, u) in enumerate(pairs_to_compare):
        if v not in f_seen:
            f_seen[v] = [i, i]
        else:
            f_seen[v][1] = i
        if u not in f_seen:
            f_seen[u] = [i, i]
        else:
            f_seen[u][1] = i
    return f_seen


def get_touch(f_seen, v, u, i):
    info = [{0: False, 1: False}, {0: False, 1: False}]
    if i == f_seen[v][0]:
        info[0][0] = True  # means to fill

    if i == f_seen[u][0]:
        info[0][1] = True  # means to fill

    if i == f_seen[v][1]:
        info[1][0] = True  # means to remove aux column

    if i == f_seen[u][1]:
        info[1][1] = True  # means to remove aux column

    return info


@task(returns=1)
def partial_sort(data, settings):
    cols = settings['columns']
    order = settings['ascending']

    data = sorting(data, cols, order)
    data.reset_index(drop=True, inplace=True)

    return data


def prepare_data(data1, data2, max_partition, info):

    n1, n2 = len(data1), len(data2)
    if info[0][0]:

        if n1 != max_partition:
            # fill data1 by 1
            data1 = data1.append(data1.iloc[-1], ignore_index=True)
            val = np.ones(n1+1, dtype=bool)
            val[-1] = False
        else:
            val = np.ones(n1, dtype=bool)

        data1['tmp_sort_idx'] = val

    if info[0][1]:
        if n2 != max_partition:
            # fill data2 by 1
            data2 = data2.append(data2.iloc[-1], ignore_index=True)
            val = np.ones(n2 + 1, dtype=bool)
            val[-1] = False
        else:
            val = np.ones(n2, dtype=bool)

        data2['tmp_sort_idx'] = val

    return data1, data2


def clean_data(data1, data2, info):
    col = 'tmp_sort_idx'

    if info[1][0] and col in data1:
        data1 = data1[data1[col] == True]
        data1.drop([col], axis=1, inplace=True)

    if info[1][1] and col in data2:
        data2 = data2[data2[col] == True]
        data2.drop([col], axis=1, inplace=True)

    return data1, data2


@constraint(ComputingUnits="2")  # approach to have more available memory
@task(returns=2)  # @task(data1=INOUT, data2=INOUT)
def _mergesort_bit(data1, data2, settings, info):
    """Return 1 if [data1, data2] is sorted, otherwise is -1.

    Space complexity: 2(N+M)
    """
    print("Time_init: {}".format(datetime.datetime.now()))
    t1 = time.time()
    cols = settings['columns']
    order = settings['ascending']
    max_rows = settings['max_partition']

    data1, data2 = prepare_data(data1, data2, max_rows, info)

    n1 = len(data1)

    data = _concatenate(data1, data2)
    del data1, data2

    data = sorting(data, cols, order)

    data1, data2 = split_df(data, n1)
    # data1.iloc[:] = data.iloc[:n1].values
    # data2.iloc[:] = data.iloc[n1:].values
    data1, data2 = clean_data(data1, data2, info)

    t2 = time.time()
    print("Time inside mergesort_bit: ", t2-t1)

    return data1, data2


@constraint(ComputingUnits="2")
@task(returns=3)
def _mergesort(data1, data2, settings):
    """Return 1 if [data1, data2] is sorted, otherwise is -1.

    Space complexity: 2(N+M)
    """
    cols = settings['columns']
    order = settings['ascending']+[True]
    cols_all = data1.columns
    n1, n2 = len(data1), len(data2)

    signal = True
    if n1 == 0 or n2 == 0:
        if n1 != 0:
            data1 = sorting(data1, cols, order)
            data1.reset_index(drop=True, inplace=True)
        elif n2 != 0:
            data2 = sorting(data2, cols, order)
            data2.reset_index(drop=True, inplace=True)

        return data1, data2, signal

    data = _concatenate(data1, data2)

    data, tmp_col, cols, ids = _adding_aux_column(data, cols, cols_all)

    data = sorting(data, cols, order)
    ids_sort = data[tmp_col].values.tolist()
    data = data.drop(columns=[tmp_col])

    data1, data2 = split_df(data, n1)

    if ids[:n1] != ids_sort[:n1]:
        signal = False

    elif ids[n1:] != ids_sort[n1:]:
        signal = False

    return data1, data2, signal


def check_column(cols):
    base = 'tmp-sort'
    i = 0
    col = '{}_{}'.format(base, i)
    while col in cols:
        col = '{}_{}'.format(base, i)
    return col


def _adding_aux_column(data, cols, cols_all):
    tmp_col = check_column(cols_all)
    # To check if has any change
    n = len(data)
    ids = np.arange(n, dtype=int).tolist()
    data[tmp_col] = ids
    cols.append(tmp_col)
    return data, tmp_col, cols, ids


def sorting(df, cols, ascending):
    t1 = time.time()

    # approach 1
    # df.sort_values(cols, inplace=True, ascending=ascending, kind='mergesort')

    # approach 2 (https://github.com/pandas-dev/pandas/issues/17111)
    for col, asc in zip(reversed(cols), reversed(ascending)):
        df.sort_values(col, inplace=True, ascending=asc, kind='mergesort')

    #  approach 3
    # ascending = ascending[0]
    # order = np.lexsort(
    #         [df[col].values for col in reversed(cols)])
    #
    # if not ascending:
    #     order = order[::-1]
    #
    # for col in list(df.columns):
    #     df[col] = df[col].values[order]
    #
    # del order

    t2 = time.time()
    print("Time inside sorting: ", t2-t1)
    return df


def split_df(data, n1):
    t1 = time.time()
    # df1, df2 = np.split(data, [n1]) # ~2x slower
    df1 = data.iloc[:n1]
    df2 = data.iloc[n1:]
    del data

    df2.reset_index(drop=True, inplace=True)

    t2 = time.time()
    print("Time inside split_df: ", t2-t1)
    return df1, df2


def _concatenate(data1, data2):
    t1 = time.time()
    data = pd.concat([data1, data2], sort=False, ignore_index=True)
    t2 = time.time()
    print("Time inside concatenate: ", t2-t1)
    return data






