#!/usr/bin/python
# -*- coding: utf-8 -*-
"""AddColumns Operation: Merge two dataframes, column-wise."""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np
import pandas as pd


def AddColumnsOperation(df1, df2, balanced, sufixes, numFrag):
    """AddColumnsOperation.

    :param df1:         A list with numFrag pandas's dataframe;
    :param df2:         A list with numFrag pandas's dataframe;
    :param balanced:    True only if len(df1[i]) == len(df2[i]) to each i;
    :param numFrag:     The number of fragments;
    :param suffixes     Suffixes for attributes (a list with 2 values);
    :return:            Returns a list with numFrag pandas's dataframe.
    """
    indexes1, indexes2, len1, len2 = check_indexes(df1, df2, numFrag)

    up1 = indexes1[-1][1]
    up2 = indexes2[-1][1]
    op = (up2 < up1)
    result = [pd.DataFrame([]) for f in range(numFrag)]
    startfill = -1

    if op:
        for f in range(numFrag):
            for j in range(numFrag):
                if is_overlapping(indexes1[f], indexes2[j]):
                    last = (f == numFrag-1) or (j == numFrag-1)
                    result[f] = AddColumns_part(result[f], df1[f], df2[j],
                                                indexes1[f], indexes2[j],
                                                sufixes, last, op)
                    startfill = f

        if startfill < numFrag-1:
            for f in xrange(startfill+1, numFrag):
                result[f] = AddColumns_fill(df1[f], len2, op, sufixes)
    else:
        for f in range(numFrag):
            for j in range(numFrag):
                if is_overlapping(indexes1[j], indexes2[f]):
                    last = (f == numFrag-1) or (j == numFrag-1)
                    result[f] = AddColumns_part(result[f], df1[j], df2[f],
                                                indexes1[j], indexes2[f],
                                                sufixes, last, op)
                    startfill = f

        if startfill < numFrag-1:
            for f in xrange(startfill+1, numFrag):
                result[f] = AddColumns_fill(df2[f], len1, op, sufixes)

    return result


def check_indexes(df1, df2, numFrag):
    """Retrieve the indexes of each fragment."""
    # first: check len of each frag
    indexes1 = [[] for f in range(numFrag)]
    indexes2 = [[] for f in range(numFrag)]
    len1 = [len_count(df1[f]) for f in range(numFrag)]
    len1 = mergeReduce(mergeCount, len1)
    len2 = [len_count(df2[f]) for f in range(numFrag)]
    len2 = mergeReduce(mergeCount, len2)

    from pycompss.api.api import compss_wait_on
    len1 = compss_wait_on(len1)
    len2 = compss_wait_on(len2)

    for f in range(numFrag):
        indexes1[f] = reindexing(len1, f)
        indexes2[f] = reindexing(len2, f)
    return indexes1, indexes2, len1, len2


@task(returns=list)
def AddColumns_part(c, a, b,  indexes1, indexes2, suffixes, last, side):
    """Peform a partial add columns."""
    if len(suffixes) == 0:
        suffixes = ('_x', '_y')

    a.index = np.arange(indexes1[0], indexes1[1])
    b.index = np.arange(indexes2[0], indexes2[1])

    if not last:
        tmp = pd.merge(a, b, left_index=True, right_index=True,
                       how='inner', suffixes=suffixes)
    else:
        if side:
            tmp = pd.merge(a, b, left_index=True, right_index=True,
                           how='left', suffixes=suffixes)
        else:
            tmp = pd.merge(a, b, left_index=True, right_index=True,
                           how='right', suffixes=suffixes)

    if len(c) != 0:
        tmp = pd.concat((c, tmp))
        tmp = tmp.groupby(tmp.index).first()

    return tmp


@task(returns=list)
def AddColumns_fill(df1, info2, op, suffixes):
    """Fill the columns that is missing."""
    cols2 = info2[2]
    cols1 = df1.columns
    if len(suffixes) == 0:
        suffixes = ('_x', '_y')
    cols1 = ['{}{}'.format(col, suffixes[0])
             if col in cols2 else col for col in cols1]
    cols2 = ['{}{}'.format(col, suffixes[1])
             if col in df1.columns else col for col in cols2]

    if op:
        df1.columns = cols1
        for col in cols2:
            df1[col] = np.nan
    else:
        df1.columns = cols2
        for col in cols1:
            df1[col] = np.nan
    return df1


def is_overlapping(indexes1, indexes2):
    """Check if the both intervals are overlapping."""
    x1, x2 = indexes1
    y1, y2 = indexes2
    over = max([x2, y2]) - min([x1, y1]) < ((x2 - x1) + (y2 - y1))
    return over


def reindexing(len1, index):
    """Create the new index interval."""
    i_start = sum(len1[1][:index])
    i_end = i_start + len1[1][index]
    return [i_start, i_end]


@task(returns=list)
def len_count(df1):
    """Count the length of each fragment."""
    col = list(df1.columns)
    return [len(df1), [len(df1)], col]


@task(returns=list)
def mergeCount(len1, len2):
    """Merge count of both fragments."""
    return [len1[0]+len2[0], len1[1]+len2[1], len1[2]]
