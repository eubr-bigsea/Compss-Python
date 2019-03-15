#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import pandas as pd


def subtract(data1, data2):
    """
    Returns a new set with containing rows in the first frame but not
     in the second one. This is equivalent to EXCEPT DISTINCT in SQL.


    :param data1: A list of pandas's DataFrame;
    :param data2: The second list of pandas's DataFrame;
    :return: A list of pandas's DataFrame.
    """

    from .distinct import distinct
    result = distinct(data1, [])
    info = result['info']
    result = result['data']

    nfrag = len(result)

    for f1 in range(nfrag):
        for df2 in data2:
            result[f1], info[f1] = _difference(result[f1], df2)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@task(returns=2)
def _difference(df1, df2):
    """Perform a Difference partial operation."""

    if len(df1) > 0:
        if len(df2) > 0:
            names = df1.columns.tolist()
            df1 = pd.merge(df1, df2, indicator=True, how='left', on=names)
            df1 = df1.loc[df1['_merge'] == 'left_only', names]

    info = [df1.columns.tolist(), df1.dtypes.values, [len(df1)]]
    return df1, info


def except_all(data1, data2):
    """
    Return a new DataFrame containing rows in this DataFrame but not in
    another DataFrame while preserving duplicates. This is equivalent to EXCEPT
    ALL in SQL.

    :param data1: A list of pandas's DataFrame;
    :param data2: The second list of pandas's DataFrame;
    :return: A list of pandas's DataFrame.
    """

    from .aggregation import AggregationOperation

    settings_agg = {'groupby': '*',
                    'operation': {'*': ['count']},
                    'aliases': {'*': ['tmp_except_all']}}

    agg_data1 = AggregationOperation().transform(data1, settings_agg)['data']
    agg_data2 = AggregationOperation().transform(data2, settings_agg)['data']

    nfrag1 = len(agg_data1)
    nfrag2 = len(agg_data2)
    info = [{} for _ in range(nfrag1)]

    for f1 in range(nfrag1):
        for f2 in range(nfrag2):
            end = f2 == (nfrag2 - 1)
            agg_data1[f1], info[f1] = _except(agg_data1[f1], agg_data2[f2], end)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': agg_data1, 'info': info}
    return output


@task(returns=2)
def _except(df1, df2, last):
    """Peform a Difference partial operation keeping duplicated rows."""

    name1, name2 = list(df1.columns), list(df2.columns)
    if len(df1) > 0 and len(df2) > 0:

        check_cols = all([True for col in name1 if col in name2])
        if len(name1) == len(name2) and check_cols:
            name1.remove('tmp_except_all')
            df1 = df1.set_index(name1).subtract(df2.set_index(name1),
                                                fill_value=0)
            df1.reset_index(inplace=True)
            df1 = df1.loc[df1['tmp_except_all'] > 0]

    if last:

        values = df1['tmp_except_all'].values
        print values

        for i, v in enumerate(values):
            for _ in range(int(v)-1):
                nfrag = len(df1)
                df1.loc[nfrag+i] = df1.loc[i]
        df1 = df1.drop(['tmp_except_all'], axis=1)

    info = [df1.columns.tolist(), df1.dtypes.values, [len(df1)]]
    return df1, info



