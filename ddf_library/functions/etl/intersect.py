#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from .hash_partitioner import hash_partition

from pycompss.api.task import task

import pandas as pd
import numpy as np


def intersect(data1, data2, settings):
    """
    Returns a new DataFrame containing rows in both frames.

    :param data1: A list with nfrag pandas's DataFrame;
    :param data2: Other list with nfrag pandas's DataFrame;
    :param settings: A dictionary with:
        - distinct: True to be equivalent to INTERSECT, False to INTERSECT ALL;
    :return: Returns a new pandas DataFrame

    .. note:: Rows with NA elements will not be take in count.
    """

    data1, data2, _ = intersect_stage_1(data1, data2, settings)

    nfrag = len(data1)
    result = [[] for _ in range(nfrag)]
    info = result[:]

    for f in range(nfrag):
        settings['id_frag'] = f
        result[f], info[f] = intersect_stage_2(data1[f], data2[f], settings)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def intersect_stage_1(data1, data2, settings):

    info1, info2 = settings['info']
    nfrag = len(data1)

    params_hash1 = {'columns': [], 'info': [info1], 'nfrag': nfrag}
    params_hash2 = {'columns': [], 'info': [info2], 'nfrag': nfrag}

    out1 = hash_partition(data1, params_hash1)
    out2 = hash_partition(data2, params_hash2)

    data1 = out1['data']
    data2 = out2['data']

    return data1, data2, settings


def intersect_stage_2(df1, df2, settings):
    """Perform a partial intersection."""

    remove_duplicates = settings.get('distinct', True)
    frag = settings['id_frag']

    keys = df1.columns.tolist()
    keys2 = df2.columns.tolist()

    df1 = df1.dropna(axis=0, how='any')

    if remove_duplicates:
        df1 = df1.drop_duplicates(subset=keys)

    df2 = df2.dropna(axis=0, how='any').drop_duplicates(subset=keys2)

    if set(keys) == set(keys2) and len(df1) > 0:
        df1 = pd.merge(df1, df2, how='inner', on=keys, copy=False)
        df1 = df1.infer_objects()

    info = generate_info(df1, frag)
    return df1, info


@task(returns=2)
def task_intersect_stage_2(df1, df2, settings):
    return intersect_stage_2(df1, df2, settings)