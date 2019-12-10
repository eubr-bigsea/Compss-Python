#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, merge_schema, merge_reduce, \
    create_stage_files, read_stage_file, save_stage_file

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.parameter import FILE_IN, FILE_OUT

import functools
import pandas as pd
import numpy as np

#TODO pode ser usado optimizacao em hash_partition


# def aggregation(data, settings):
#     """
#     Computes aggregates and returns the result as a DataFrame
#
#     :param data: A list with nfrag pandas's DataFrame;
#     :param settings: A dictionary that contains:
#         - groupby: A list with the columns names to aggregates;
#         - aliases: A dictionary with the aliases of all aggregated columns;
#         - operation: A dictionary with the functions to be applied in
#                      the aggregation:
#             'mean': Computes average values for each numeric columns
#              for each group;
#             'count': Counts the number of records for each group;
#             'first': Returns the first element of group;
#             'last': Returns the last element of group;
#             'max': Computes the max value for each numeric columns;
#             'min': Computes the min value for each numeric column;
#             'sum': Computes the sum for each numeric columns for each group;
#             'list': Returns a list of objects with duplicates;
#             'set': Returns a set of objects with duplicate elements
#              eliminated.
#     :return: Returns a list of pandas's DataFrame.
#
#     example:
#         settings['groupby']   = ["col1"]
#         settings['operation'] = {'col2':['sum'],'col3':['first','last']}
#         settings['aliases']   = {'col2':["Sum_col2"],
#                                  'col3':['col_First','col_Last']}
#
#     """
#
#     data, _ = aggregation_stage_1(data, settings)
#     nfrag = len(data)
#
#     # 3º perform a global aggregation
#     result = create_stage_files(settings['stage_id'], nfrag)
#     info = result[:]
#
#     for f in range(nfrag):
#         settings['id_frag'] = f
#         result[f], info[f] = task_aggregation_stage_2(data[f], settings.copy())
#
#     compss_delete_object(data)
#     output = {'key_data': ['data'], 'key_info': ['info'],
#               'data': result, 'info': info}
#     return output


def aggregation_stage_1(data, settings):
    nfrag = len(data)

    # 1º perform a local aggregation in each partition
    partial_agg = create_stage_files(settings['stage_id'], nfrag)
    info = [[] for _ in range(nfrag)]
    for f in range(nfrag):
        info[f] = _aggregate(data[f], partial_agg[f], settings, f)

    # 2º perform a hash partition
    info = merge_reduce(merge_schema, info)
    info = compss_wait_on(info)

    from .hash_partitioner import hash_partition
    params = {'nfrag': nfrag,
              'columns': settings['groupby'],
              'info': [info],
              'stage_id': settings['stage_id']}

    data = hash_partition(partial_agg, params)['data']
    compss_delete_object(partial_agg)

    return data, settings


@task(returns=1, data_in=FILE_IN, data_out=FILE_OUT)
def _aggregate(data_in, data_out, params, f):
    """Perform a partial aggregation."""
    columns = params['groupby']
    operation_list = params['operation']
    data = read_stage_file(data_in)

    if columns is '*':
        columns = list(data.columns)

    operations = _generate_agg_operations(operation_list, columns)
    data = data.groupby(columns).agg(**operations)\
        .reset_index()\
        .reset_index(drop=True)
    info = generate_info(data, f)
    save_stage_file(data_out, data)
    return info


def aggregation_stage_2(data1, params):
    """Combining the aggregation with other fragment.

    if a key is present in both fragments, it will remain
    in the result only if f1 <f2.
    """
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
