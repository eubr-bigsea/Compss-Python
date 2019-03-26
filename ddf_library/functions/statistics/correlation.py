#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.local import local

import numpy as np


def correlation(data, settings):
    """
    Calculate the Pearson Correlation Coefficient.

    :param data: A list of pandas's DataFrame;
    :param settings: A dictionary that contains:
        - 'col1': The name of the first column
        - 'col2': The name of the second column

    :return: A list of pandas's DataFrame;
    """

    info = [_covariance_stage1(df, settings) for df in data]
    agg_info = merge_reduce(_covariance_stage2, info)

    error = [_covariance_stage3(df, agg_info) for df in data]
    error = merge_reduce(_describe_stage4, error)

    result = _generate_stage5(agg_info, error)

    return result


@task(returns=1)
def _covariance_stage1(df, settings):
    col1 = settings['col1']
    col2 = settings['col2']
    columns = [col1, col2]

    sums = df[columns].sum(skipna=True, numeric_only=True).values
    count = len(df)
    if len(sums) == 0:
        sums = [0, 0]
    info = {'columns': columns, 'count': count, 'sum': sums}

    return info


@task(returns=1)
def _covariance_stage2(info1, info2):

    count = info1['count'] + info2['count']
    sum_values = np.add(info1['sum'], info2['sum'])

    info = {'columns': info1['columns'], 'count': count, 'sum': sum_values}

    return info


@task(returns=1)
def _covariance_stage3(df, info_agg):
    # generate error

    mean = [np.divide(float(m), info_agg['count']) for m in info_agg['sum']]
    col1, col2 = info_agg['columns']

    error1 = df[col1].values - mean[0]
    error2 = df[col2].values - mean[1]
    error = (np.array(error1)*np.array(error2)).sum()
    sse1 = (error1 ** 2).sum()
    sse2 = (error2 ** 2).sum()

    return [error, sse1, sse2]


@task(returns=1)
def _describe_stage4(info1, info2):
    error1_1, sse1_1, sse2_1 = info1
    error1_2, sse1_2, sse2_2 = info2

    error1 = np.add(error1_1, error1_2)
    sse1 = np.add(sse1_1, sse1_2)
    sse2 = np.add(sse2_1, sse2_2)
    return [error1, sse1, sse2]


@local
def _generate_stage5(agg_info, info):
    error, sse1, sse2 = info
    count = agg_info['count']
    std1 = np.sqrt(np.divide(float(sse1), count - 1))
    std2 = np.sqrt(np.divide(float(sse2), count - 1))

    if std1 == 0 or std2 == 0:
        corr = np.nan
    else:

        cov = np.divide(float(error), count-1)
        corr = round(np.divide(cov, std1*std2), 5)

    return corr
