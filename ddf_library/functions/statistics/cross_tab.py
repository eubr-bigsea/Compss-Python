#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.local import local

import numpy as np
import pandas as pd


def cross_tab(data, settings):
    """
    Computes a pair-wise frequency table of the given columns. Also known as
    a contingency table. The number of distinct values for each column should
    be less than 1e4. At most 1e6 non-zero pair frequencies will be returned.

    :param data: A list of pandas's DataFrame;
    :param settings: A dictionary that contains:
        - 'col1': The name of the first column
        - 'col2': The name of the second column
    :return: A list of pandas's DataFrame;
    """

    col1 = settings['col1']
    col2 = settings['col2']
    cols = [col1, col2]
    nfrag = len(data)

    for f in range(nfrag):
        data[f] = _crosstab_partial(data[f], cols)

    data = merge_reduce(_merge_counts, data)
    data, info = _create_tab(data, nfrag)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': data, 'info': info}
    return output


@task(returns=1)
def _crosstab_partial(data, cols):
    col1, col2 = cols
    data = pd.crosstab(index=data[col1], columns=data[col2])
    data.columns = data.columns.values

    data.index = data.index.values
    return data


@task(returns=1)
def _merge_counts(data1, data2):

    data = data1.add(data2, fill_value=0).fillna(0).astype(int)
    size_cols = data.shape[1]
    size_len = data.shape[0]
    if size_cols > 1e4:
        data = data.drop(data.columns[10000:], axis=1)
    if size_len > 1e6:
        data = data[:1000000]
    return data


@local
def _create_tab(data, nfrag):
    data.insert(0, 'key', data.index.values)

    cols = data.columns.values
    dtypes = data.dtypes.values

    data = np.array_split(data, nfrag)
    info = [[cols, dtypes, [d]] for d in data]

    return data, info
