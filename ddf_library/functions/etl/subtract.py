#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from pycompss.api.task import task
import pandas as pd


def subtract(data1, data2, settings):
    """
    Returns a new set with containing rows in the first frame but not
    in the second one. This is equivalent to EXCEPT DISTINCT in SQL.

    :param data1: A list of pandas's DataFrame;
    :param data2: The second list of pandas's DataFrame;
    :param settings: A dictionary with:
    :return: A list of pandas's DataFrame.
    """
    data1, data2, _ = subtract_stage_1(data1, data2, settings)

    nfrag = len(data1)
    result = [[] for _ in range(nfrag)]
    info = result[:]

    for f in range(nfrag):
        settings['id_frag'] = f
        result[f], info[f] = task_subtract_stage_2(data1[f], data2[f],
                                                   settings.copy())

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def subtract_stage_1(data1, data2, settings):

    info1, info2 = settings['info']
    nfrag = len(data1)

    from .distinct import distinct
    params = {'columns': [], 'info': [info1]}
    out1 = distinct(data1, params)
    data1 = out1['data']

    from .hash_partitioner import hash_partition
    params_hash2 = {'columns': [], 'info': [info2], 'nfrag': nfrag}
    out2 = hash_partition(data2, params_hash2)
    data2 = out2['data']

    return data1, data2, settings


def subtract_stage_2(df1, df2, settings):
    """Perform a Difference partial operation."""
    frag = settings['id_frag']

    if len(df1) > 0:
        if len(df2) > 0:
            names = df1.columns.tolist()
            df1 = pd.merge(df1, df2, indicator=True, how='left', on=names)
            df1 = df1.loc[df1['_merge'] == 'left_only', names]

            df1 = df1.infer_objects()

    info = generate_info(df1, frag)
    return df1, info


@task(returns=2)
def task_subtract_stage_2(df1, df2, settings):
    return subtract_stage_2(df1, df2, settings)
