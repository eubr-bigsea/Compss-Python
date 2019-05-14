#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, merge_info

from pycompss.api.task import task

import pandas as pd
import numpy as np
import time


def sort(data, settings):
    """
    Returns a DataFrame sorted by the specified column(s).

    :param data: A list of pandas's DataFrame;
    :param settings: A dictionary that contains:
        - columns: The list of columns to be sorted.
        - ascending: A list indicating whether the sort order
            is ascending (True) for each column.
    :return: A list of pandas's DataFrame
    """

    data, settings = sort_stage_1(data, settings)

    nfrag = len(data)
    result = [[] for _ in range(nfrag)]
    info = result[:]

    for f in range(nfrag):
        settings['id_frag'] = f
        result[f], info[f] = task_sort_stage_2(data[f], settings)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def sort_stage_1(data, settings):

    nfrag = len(data)
    settings = preprocessing(settings)
    info = settings['info'][0]

    if nfrag > 1:
        from .range_partitioner import range_partition
        params = {'nfrag': nfrag,
                  'columns': settings['columns'],
                  'ascending': settings['ascending'],
                  'info': [info]}
        output_range = range_partition(data, params)
        data = output_range['data']

    return data, settings


def preprocessing(settings):
    """Check all settings."""
    columns = settings.get('columns', [])
    if not isinstance(columns, list):
        columns = [columns]

    if len(columns) == 0:
        raise Exception('`columns` do not must be empty.')

    asc = settings.get('ascending', [])
    if not isinstance(asc, list):
        asc = [asc]

    n1, n2 = len(columns), len(asc)
    if n1 > n2:
        asc = asc + [True for _ in range(n2-n1)]
    elif n2 > n1:
        asc = asc[:n1]

    settings['columns'] = columns
    settings['ascending'] = asc
    return settings


def sort_stage_2(data, settings):
    cols = settings['columns']
    order = settings['ascending']
    frag = settings['id_frag']

    data.sort_values(cols, inplace=True, ascending=order)
    data.reset_index(drop=True, inplace=True)

    info = generate_info(data, frag)

    return data, info


@task(returns=2)
def task_sort_stage_2(data, settings):
    return sort_stage_2(data, settings)
