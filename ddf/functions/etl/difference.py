#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd


def difference(data1, data2):
    """
    Returns a new set with containing rows in the first frame but not
     in the second one.

    :param data1: A list with nfrag pandas's dataframe;
    :param data2: The second list with nfrag pandas's dataframe;

    :return: A list with nfrag pandas's dataframe.

    """
    if isinstance(data1[0], pd.DataFrame):
        # it is necessary to perform a deepcopy if data is not a
        # FutureObject
        # to enable multiple branches executions
        import copy
        result = copy.deepcopy(data1)
    else:
        result = data1[:]

    nfrag = len(data1)
    info = [[] for _ in range(nfrag)]

    for f1 in range(nfrag):
        for df2 in range(data2):
            result[f1], info[f1] = _difference(result[f1], df2)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output
    

@task(returns=2)
def _difference(df1, df2):
    """Peform a Difference partial operation."""
    if len(df1) > 0:
        if len(df2) > 0:
            names = df1.columns
            df1 = pd.merge(df1, df2, indicator=True, how='left', on=None)
            df1 = df1.loc[df1['_merge'] == 'left_only', names]

    info = [df1.columns.tolist(), df1.dtypes.values, [len(df1)]]
    return df1, info



