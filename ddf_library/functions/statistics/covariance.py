#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_delete_object, compss_wait_on

from ddf_library.utils import read_stage_file

import numpy as np


def covariance(data, settings):
    """
    Calculate the sample covariance for the given columns, specified by their
    names, as a double value.

    :param data: A list of pandas's DataFrame;
    :param settings: A dictionary that contains:
        - 'col1': The name of the first column
        - 'col2': The name of the second column

    :return: A list of pandas's DataFrame;
    """

    info = [_covariance_stage1(df, settings) for df in data]
    agg_info = merge_reduce(_covariance_stage2, info)
    compss_delete_object(info)

    error = [_covariance_stage3(df, agg_info) for df in data]
    agg_error = merge_reduce(_describe_stage4, error)
    compss_delete_object(error)

    result = _generate_stage5(agg_info, agg_error)

    return result


@task(returns=1, data_input=FILE_IN)
def _covariance_stage1(data_input, settings):
    col1, col2 = settings['col1'], settings['col2']
    columns = [col1, col2]
    df = read_stage_file(data_input, columns)

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


@task(returns=1, data_input=FILE_IN)
def _covariance_stage3(data_input, info_agg):
    # generate error

    mean = [np.divide(float(m), info_agg['count']) for m in info_agg['sum']]
    col1, col2 = info_agg['columns']
    df = read_stage_file(data_input, info_agg['columns'])

    error1 = df[col1].values - mean[0]
    error2 = df[col2].values - mean[1]
    error = (error1*error2).sum()

    return error


@task(returns=1)
def _describe_stage4(error1, error2):
    error1 = np.add(error1, error2)
    return error1


# @local
def _generate_stage5(agg_info, error):
    agg_info = compss_wait_on(agg_info)
    error = compss_wait_on(error)
    count = agg_info['count']

    if count == 1:
        raise Exception("Number of samples is too small.")

    cov = round(error / (count-1), 5)
    return cov
