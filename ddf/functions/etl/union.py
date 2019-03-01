#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd


def union(data1, data2):
    """
    Function which do a union between two pandas DataFrame.

    :param data1:   A list with nfrag pandas's dataframe;
    :param data2:   Other list with nfrag pandas's dataframe;
    :return:        Returns a list with nfrag pandas's dataframe.

    :note: Need schema as input
    """

    nfrag = len(data1)
    result = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]
    for f in range(nfrag):
        result[f] = _union(data1[f], data2[f])

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@task(returns=2)
def _union(df1, df2):
    """Perform a partil union."""

    if len(df1) == 0:
        result = df2
    elif len(df2) == 0:
        result = df1
    else:
        result = pd.concat([df1, df2], ignore_index=True)

    info = [result.columns.tolist(), result.dtypes.values, [len(result)]]
    return result, info

