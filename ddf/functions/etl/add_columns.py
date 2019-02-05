#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.local import local
from pycompss.functions.reduce import mergeReduce
import numpy as np
import pandas as pd


class AddColumnsOperation(object):
    """AddColumns Operation: Merge two dataframes, column-wise."""

    def transform(self, df1, df2, sufixes, nfrag):
        """AddColumnsOperation.

        :param df1: A list with nfrag pandas's dataframe;
        :param df2: A list with nfrag pandas's dataframe;
        :param nfrag: The number of fragments;
        :param sufixes: Suffixes for attributes (a list with 2 values);
        :return: Returns a list with nfrag pandas's dataframe.
        """
        result = [pd.DataFrame([]) for _ in range(nfrag)]
        info1 = [len_count(df1[f]) for f in range(nfrag)]
        info2 = [len_count(df2[f]) for f in range(nfrag)]
        len1 = mergeReduce(_merge_count, info1)
        len2 = mergeReduce(_merge_count, info2)

        idxs1, idxs2, len1, len2 = _check_indexes(len1, len2, nfrag)
        up1 = idxs1[-1][1]
        up2 = idxs2[-1][1]
        op = (up2 < up1)
        startfill = -1

        if op:
            for f in range(nfrag):
                for j in range(nfrag):
                    if _is_overlapping(idxs1[f], idxs2[j]):
                        last = (f == nfrag-1) or (j == nfrag-1)
                        result[f] = _add_columns(result[f], df1[f],
                                                 df2[j], idxs1[f],
                                                 idxs2[j], sufixes,
                                                 last, op)
                        startfill = f

            if startfill < nfrag-1:
                for f in xrange(startfill+1, nfrag):
                    result[f] = _add_columns_fill(df1[f], len2,
                                                       op, sufixes)
        else:
            for f in range(nfrag):
                for j in range(nfrag):
                    if _is_overlapping(idxs1[j], idxs2[f]):
                        last = (f == nfrag-1) or (j == nfrag-1)
                        result[f] = _add_columns(result[f], df1[j],
                                                 df2[f], idxs1[j],
                                                 idxs2[f], sufixes,
                                                 last, op)
                        startfill = f

            if startfill < nfrag-1:
                for f in xrange(startfill+1, nfrag):
                    result[f] = _add_columns_fill(df2[f], len1,
                                                  op, sufixes)

        return result


@task(returns=list)
def len_count(df1):
    """Count the length of each fragment."""
    col = list(df1.columns)
    return [len(df1), [len(df1)], col]


@task(returns=list)
def _merge_count(len1, len2):
    """Merge count of both fragments."""
    return[len1[0] + len2[0], len1[1] + len2[1], len1[2]]

@local
def _check_indexes(len1, len2, nfrag):
    """Retrieve the indexes of each fragment."""
    indexes1 = [[] for _ in range(nfrag)]
    indexes2 = [[] for _ in range(nfrag)]

    for f in range(nfrag):
        indexes1[f] = _reindexing(len1, f)
        indexes2[f] = _reindexing(len2, f)
    return indexes1, indexes2, len1, len2


def _is_overlapping(indexes1, indexes2):
    """Check if the both intervals are overlapping."""
    x1, x2 = indexes1
    y1, y2 = indexes2
    over = max([x2, y2]) - min([x1, y1]) < ((x2 - x1) + (y2 - y1))
    return over


def _reindexing(len1, index):
    """Create the new index interval."""
    len1 = len1[1]
    i_start = sum(len1[:index])
    i_end = i_start + len1[index]
    return [i_start, i_end]


@task(returns=list)
def _add_columns(c, a, b,  idxs1, idxs2, suffixes, last, side):
    """Peform a partial add columns."""
    if len(suffixes) == 0:
        suffixes = ('_x', '_y')

    a.index = np.arange(idxs1[0], idxs1[1])
    b.index = np.arange(idxs2[0], idxs2[1])

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
def _add_columns_fill(df1, info2, op, suffixes):
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
