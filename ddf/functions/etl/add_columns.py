#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.local import local
from pycompss.functions.reduce import merge_reduce
import numpy as np
import pandas as pd


class AddColumnsOperation(object):

    @staticmethod
    def transform(df1, df2, suffixes=['_l', '_r']):
        """
        Merge two DataFrames, column-wise.

        :param df1: A list with nfrag pandas's DataFrame;
        :param df2: A list with nfrag pandas's DataFrame;
        :param suffixes: Suffixes for attributes (a list with 2 values);
        :return: Returns a list with nfrag pandas's dataframe.

        :note: Need schema as input
        """
        nfrag1 = len(df1)
        nfrag2 = len(df2)

        # Getting information about the schema and number of rows
        info1 = [len_count(df1[f]) for f in range(nfrag1)]
        info2 = [len_count(df2[f]) for f in range(nfrag2)]
        len1 = merge_reduce(_merge_count, info1)
        len2 = merge_reduce(_merge_count, info2)
        len1, len2 = _check_indexes(len1, len2)

        # The larger dataset will be always df1
        swap_cols = False
        if len2[0] > len1[0]:
            len1, len2 = len2, len1
            df1, df2 = df2, df1
            nfrag1, nfrag2 = nfrag2, nfrag1
            swap_cols = True

        if swap_cols:
            new_df2 = [pd.DataFrame([], columns=len2[2]) for _ in range(nfrag1)]
        else:
            new_df2 = [pd.DataFrame([], columns=len1[2]) for _ in range(nfrag1)]

        # re-organizing df2 to have df1's shape
        ranges2 = [[0, end] for end in len2[1]]
        extract_from_df2 = {}
        for i, left in enumerate(len1[1]):
            extract_from_df2[i] = [0 for _ in range(nfrag2)]

            for j, (start, end) in enumerate(ranges2):
                if left <= 0:
                    break
                size2 = (end - start)
                if size2 != 0:
                    if left < size2:
                        extract_from_df2[i][j] = [start, start+left]
                        ranges2[j] = [start+left, start+end]
                        left = 0
                    else:
                        left -= size2
                        extract_from_df2[i][j] = [start, start + size2]
                        ranges2[j] = [start + size2, start + size2]

        for f in range(nfrag1):
            for i, op in enumerate(extract_from_df2[f]):
                if op != 0:
                    new_df2[f] = _concatenate_df(new_df2[f], df2[i], op)

        del ranges2
        del extract_from_df2
        # 3º merging two DataFrames with same shape
        result = [[] for _ in range(nfrag1)]
        for f in range(nfrag1):
            result[f] = _add_columns(df1[f], new_df2[f], suffixes, swap_cols)

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
def _check_indexes(len1, len2):
    """Retrieve the indexes of each fragment."""
    return len1, len2


@task(returns=1)
def _concatenate_df(new_df, df, op):
    i1, i2 = op

    df = df.iloc[i1: i2]

    if len(new_df) > 0:
        new_df = pd.concat([new_df, df])
    else:
        new_df = df
    return new_df


@task(returns=list)
def _add_columns(df1, df2, suffixes, swap_cols):
    """Peform a partial add columns."""

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    mode = 'left'
    if swap_cols:
        df1, df2 = df2, df1
        mode = 'right'

    tmp = pd.merge(df1, df2, left_index=True, right_index=True,
                   how=mode, suffixes=suffixes)

    tmp.reset_index(drop=True, inplace=True)
    return tmp

