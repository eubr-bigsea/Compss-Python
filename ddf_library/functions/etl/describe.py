#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.local import local

import pandas as pd
import numpy as np


def describe(data, columns):
    """
    Computes basic statistics for numeric and string columns.

    This include count, number of NaN, mean, stddev, min, and max.
    :param data: A list of pandas's DataFrame;
    :param columns: A list of columns, if no columns are given,
    this function computes statistics for all numerical or string columns;
    :return:

    ..note: if we already know the size, can be better structured
    """

    if not isinstance(columns, list):
        columns = [columns]

    info = [_describe_stage1(df, columns) for df in data]
    agg_info = merge_reduce(_describe_stage2, info)

    sse = [_describe_stage3(df, agg_info) for df in data]
    sse = merge_reduce(_describe_stage4, sse)

    result = _generate_stage5(agg_info, sse, len(data))

    return result


@task(returns=1)
def _describe_stage1(df, columns):
    if len(columns) == 0:
        columns = df.columns.values

    # count, mean, stddev, min, and max
    count = len(df)
    nan = df[columns].isna().sum()
    nan = [nan.loc[att] if att in nan.index else np.nan for att in columns]
    mean = df[columns].sum(skipna=True, numeric_only=True)
    means = [mean.loc[att] if att in mean.index else np.nan for att in columns]
    minimum = df[columns].min(skipna=True, axis=0).values
    maximum = df[columns].max(skipna=True, axis=0).values

    info = {'columns': columns, 'count': count, 'mean': means,
            'min': minimum, 'max': maximum, 'nan': nan}
    return info


@task(returns=1)
def _describe_stage2(info1, info2):

    count = info1['count'] + info2['count']
    sum_values = np.add(info1['mean'], info2['mean'])
    nan = np.add(info1['nan'], info2['nan'])
    minimum = [min([x, y]) for x, y in zip(info1['min'], info2['min'])]
    maximum = [max([x, y]) for x, y in zip(info1['max'], info2['max'])]

    info = {'columns': info1['columns'], 'count': count, 'mean': sum_values,
            'min': minimum, 'max': maximum, 'nan': nan}

    return info


@task(returns=1)
def _describe_stage3(df, info_agg):
    # generate error

    mean = [m/info_agg['count'] for m in info_agg['mean']]
    columns = info_agg['columns']

    sse = []
    for i, att in enumerate(columns):
        if np.issubdtype(df[att].dtype, np.number):
            sse.append(
                    df.apply(lambda row: (row[att]-mean[i])**2, axis=1).sum())
        else:
            sse.append(np.nan)
    return sse


@task(returns=1)
def _describe_stage4(sse1, sse2):
    sse = np.add(sse1, sse2)
    return sse


@local
def _generate_stage5(agg_info, sse, nfrag):
    mean = [m / agg_info['count'] if m else np.nan for m in agg_info['mean']]

    count = agg_info['count']
    std = []
    counts = [count for _ in range(len(sse))]
    for e in sse:
        std.append(np.sqrt(float(e) / (count - 1)))

    cols = agg_info['columns'].tolist() + ['statistic']
    statistics = ['column', 'count', 'mean', 'std', 'min', 'max', 'nan_count']

    result = pd.DataFrame([counts,
                          mean,
                          std,
                          agg_info['min'],
                          agg_info['max'],
                          agg_info['nan'].tolist()],
                          columns=agg_info['columns'],
                          index=['count', 'mean', 'std', 'min', 'max',
                                 'nan_count']).T

    result.reset_index(inplace=True)
    result = result.rename(columns={'index': 'column'})

    return result
