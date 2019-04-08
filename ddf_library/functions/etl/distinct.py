#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from pycompss.api.task import task
import pandas as pd


def distinct(data, cols):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.

    :param data: A list with nfrag pandas's dataframe;
    :param cols: A list with the columns names to take in count
                (if no field is choosen, all fields are used).
    :return: Returns a list with nfrag pandas's dataframe.
    """
    nfrag = len(data)

    if isinstance(data[0], pd.DataFrame):
        # it is necessary to perform a deepcopy if data is not a FutureObject
        # to enable multiple branches executions
        import copy
        result = copy.deepcopy(data)
    else:
        result = data[:]

    info = [[] for _ in result]

    x_i, y_i = generate_pairs(nfrag)
    for f, (x, y) in enumerate(zip(x_i, y_i)):
        result[x], result[y], info[x], info[y] = \
            _drop_duplicates(result[x], result[y], cols, f)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def generate_pairs(nfrag):
    import itertools
    buff = list(itertools.combinations([x for x in range(nfrag)], 2))

    def disjoint(a, b):
        return set(a).isdisjoint(b)

    x_i = []
    y_i = []

    while len(buff) > 0:
        x = buff[0][0]
        step_list_i = []
        step_list_j = []
        if x >= 0:
            y = buff[0][1]
            step_list_i.append(x)
            step_list_j.append(y)
            buff[0] = [-1, -1]
            for j in range(len(buff)):
                tuples = buff[j]
                if tuples[0] >= 0:
                    if disjoint(tuples, step_list_i):
                        if disjoint(tuples, step_list_j):
                            step_list_i.append(tuples[0])
                            step_list_j.append(tuples[1])
                            buff[j] = [-1, -1]
        del buff[0]
        x_i.extend(step_list_i)
        y_i.extend(step_list_j)
    return x_i, y_i
        

@task(returns=4)
def _drop_duplicates(data1, data2, cols, frag):
    """Remove duplicate rows based in two fragments at the time."""
    data = pd.concat([data1, data2], axis=0, ignore_index=True, sort=False)
    data1.reset_index(drop=True, inplace=True)
    data2.reset_index(drop=True, inplace=True)

    n1 = len(data1)
    n2 = len(data2)
    if (n1+n2) > 0:
        alls_cols = data1.columns
        tmp_col = get_column(alls_cols)
        n = n1 + n2

        # Create a temporary column for referencing
        index = ['p1' for _ in range(n1)] + ['p2' for _ in range(n1, n)]
        data[tmp_col] = index

        # if cols is empty, all fields will be used)
        if len(cols) == 0:
            cols = alls_cols

        # Create a new dataFrame with only unique rows
        # Version 0.21.0 introduces the method infer_objects() for converting
        # columns of a DataFrame that have an object datatype to a more
        # specific type.
        data = data.infer_objects()\
            .drop_duplicates(cols, keep='first')\
            .reset_index(drop=True)

        # Keep rows of each dataFrame based in the temporary one
        tmp1 = data.loc[data[tmp_col] == 'p1', alls_cols].values
        m1 = len(tmp1)
        data1.iloc[0:m1, :] = tmp1
        tmp1 = data.loc[data[tmp_col] == 'p2', alls_cols].values
        m2 = len(tmp1)
        data2.iloc[0:m2, :] = tmp1

        data1.drop(data1.index[m1:], inplace=True)
        data2.drop(data2.index[m2:], inplace=True)

    info1 = generate_info(data1, frag)
    info2 = generate_info(data2, frag)

    return data1, data2, info1, info2


def get_column(cols):
    """Check a available column to use as a temporary column."""
    tmp = 'dropDup_index_0'
    i = 1
    while tmp in cols:
        tmp = 'dropDup_index_{}'.format(i)
        i += 1
    return tmp
