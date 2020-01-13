#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info

import functools
import pandas as pd
import numpy as np


def aggregation_stage_1(data, settings):
    nfrag = len(data)
    info1 = settings['info'][0]
    info1['function'] = [_aggregate, settings]

    from .hash_partitioner import hash_partition

    hash_params = {'columns': settings['groupby'],
                   'nfrag': nfrag, 'info': [info1]}
    output = hash_partition(data, hash_params)['data']

    return output, settings


def _aggregate(data, params):
    """Perform a partial aggregation."""
    columns = params['groupby']
    operation_list = params['operation']

    if columns is '*':
        columns = list(data.columns)

    operations = _generate_agg_operations(operation_list, columns)
    data = data.groupby(columns).agg(**operations)\
        .reset_index()\
        .reset_index(drop=True)

    return data, params


def aggregation_stage_2(data1, params):
    """Combining the aggregation with other fragment."""
    columns = params['groupby']
    operation_list = params['operation']
    frag = params['id_frag']

    operations = _generate_agg_operations(operation_list, columns, True)
    data1 = data1.groupby(columns).agg(**operations).reset_index()\
        .reset_index(drop=True)

    # sequence = columns + [c for c in data1.columns.tolist()
    #                       if c not in columns]
    # data1 = data1[sequence]

    info = generate_info(data1, frag)
    return data1, info


def _generate_agg_operations(operations_list, groupby, replace=False):
    """
    Used to create a pd.NamedAgg list to be used by Pandas to aggregate.
    Replace option is used in to merge the partial result of each partition.
    """
    operations = dict()
    for col, func, target in operations_list:
        if '*' in col:
            target = groupby

        if replace:
            if func in 'count':
                func = 'sum'
            elif func == 'set':
                func = _merge_set
            elif func == 'list':
                func = _merge_list
            col = target

        else:
            if func == 'list':
                func = list
            elif func == 'set':
                func = _collect_set

        operations[target] = pd.NamedAgg(column=col, aggfunc=func)
    return operations


def _merge_list(series):
    return functools.reduce(lambda x, y: list(x) + list(y),
                            series.tolist())


def _collect_set(x):
    """Generate a set of a group."""
    return list(set(x))


def _merge_set(series):
    """Merge set list."""
    return functools.reduce(lambda x, y: list(set(list(x) + list(y))),
                            series.tolist())
