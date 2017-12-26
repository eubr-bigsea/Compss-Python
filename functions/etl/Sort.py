#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Sort Operation: Returns a DataFrame sorted by the specified column(s)."""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
import pandas as pd


def SortOperation(data, settings, numFrag):
    """SortOperation.

    :param data:        A list with numFrag pandas's dataframe;
    :param settings:    A dictionary that contains:
        - algorithm:
            * 'odd-even', to sort using Odd-Even Sort (default);
            * 'bitonic',  to sort using Bitonic Sort
              (only if numFrag is power of 2);
        - columns:      The list of columns to be sorted.
        - ascending:    A list indicating whether the sort order
                        is ascending (True) for each column.
    :param numFrag:     The number of fragments;
    :return:            A list with numFrag pandas's dataframe

    Note: the list of columns should have the same size of the list
    of boolean to indicating if it is ascending sorting.
    """
    algorithm = Validate(settings, numFrag)

    if algorithm == "bitonic":
        result = sort_byBittonic(data, settings, numFrag)
    else:
        result = sort_byOddEven(data, settings, numFrag)

    return result


def Validate(settings, numFrag):
    """Check all settings."""
    cols1 = settings.get('columns', [])
    asc = settings.get('ascending', [])
    if any([len(cols1) == 0,
            len(asc) == 0,
            len(cols1) != len(asc)]):
        raise Exception('The list of `columns` ans `ascending` should have '
                        'equal lenght (and diffent form zero).')

    def is_power2(num):
        return ((num & (num - 1)) == 0) and num != 0

    algorithm = settings.get('algorithm', 'odd-even')
    if not is_power2(numFrag):
        algorithm == 'odd-even'

    return algorithm


def sort_byBittonic(data, settings, numFrag):
    """Sort by Bittonic Sort.

    Given an unordered sequence of size 2*fragments, exactly
    log2 (fragments) stages of merging are required to produce
    a completely ordered list.
    """
    data = bitonic_sort(data, settings)
    return data


def sort_byOddEven(data, settings, numFrag):
    """Sort by Odd-Even Sort."""
    for f in range(numFrag):
        data[f] = sort_p(data[f], settings)

    f = 0
    nsorted = True
    from pycompss.api.api import compss_wait_on
    while nsorted:
        signals = [0 for i in range(numFrag)]
        if (f % 2 == 0):
            for i in range(numFrag):
                if (i % 2 == 0):
                    signals[i] = mergesort(data[i], data[i+1], settings)

        else:
            for i in range(numFrag-1):
                if (i % 2 != 0):
                    signals[i] = mergesort(data[i], data[i+1], settings)

        if f > 2:
            signals = compss_wait_on(signals)
            nsorted = any([i == -1 for i in signals])
            # nsorted = False
        f += 1
    return data


def bitonic_sort(x, settings):
    """Recursive method of the Bittonic sort."""
    if len(x) <= 1:
        return x
    else:
        first = bitonic_sort(x[:len(x) // 2], settings)
        second = bitonic_sort(x[len(x) // 2:], settings)
        return bitonic_merge(first + second, settings)


def bitonic_merge(x, settings):
    """Recursive method of the Bittonic sort."""
    if len(x) == 1:
        return x
    else:
        bitonic_compare(x, settings)
        first = bitonic_merge(x[:len(x) // 2], settings)
        second = bitonic_merge(x[len(x) // 2:], settings)
        return first + second


def bitonic_compare(x, settings):
    """Sort the elements in two fragments at time."""
    dist = len(x) // 2
    for i in range(dist):
        mergesort(x[i], x[i+dist], settings)


@task(returns=list)
def sort_p(data, settings):
    """Perform a partial sort."""
    col = settings['columns']
    order = settings['ascending']
    data.sort_values(col, ascending=order, inplace=True)
    data = data.reset_index(drop=True)
    return data


@task(data1=INOUT, data2=INOUT, returns=int)
def mergesort(data1, data2, settings):
    """Return 1 if [data1, data2] is sorted, otherwise is -1."""
    col = settings['columns']
    order = settings['ascending']
    n1 = len(data1)
    n2 = len(data2)

    if n1 == 0 or n2 == 0:
        return 1

    nsorted = 1
    data = pd.concat([data1, data2])
    data.reset_index(drop=True, inplace=True)
    indexes = data.index
    data.sort_values(col, ascending=order, inplace=True)
    if any(data.index != indexes):
        nsorted = -1

    data = data.reset_index(drop=True)
    data1.ix[0:] = data.ix[:n1]

    data = data[data.index >= n1]
    data = data.reset_index(drop=True)
    data2.ix[0:] = data.ix[:]

    return nsorted
