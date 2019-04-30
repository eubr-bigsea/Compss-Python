#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, merge_schema, merge_reduce

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_delete_object

import pandas as pd


class AggregationOperation(object):
    """Computes aggregates and returns the result as a DataFrame."""

    @staticmethod
    def transform(data, settings):
        """AggregationOperation.

        :param data: A list with nfrag pandas's DataFrame;
        :param settings: A dictionary that contains:
            - groupby: A list with the columns names to aggregates;
            - aliases: A dictionary with the aliases of all aggregated columns;
            - operation: A dictionary with the functions to be applied in
                         the aggregation:
                'mean': Computes average values for each numeric columns
                 for each group;
                'count': Counts the number of records for each group;
                'first': Returns the first element of group;
                'last': Returns the last element of group;
                'max': Computes the max value for each numeric columns;
                'min': Computes the min value for each numeric column;
                'sum': Computes the sum for each numeric columns for each group;
                'list': Returns a list of objects with duplicates;
                'set': Returns a set of objects with duplicate elements
                 eliminated.
        :return: Returns a list of pandas's DataFrame.

        example:
            settings['groupby']   = ["col1"]
            settings['operation'] = {'col2':['sum'],'col3':['first','last']}
            settings['aliases']   = {'col2':["Sum_col2"],
                                     'col3':['col_First','col_Last']}

        """

        nfrag = len(data)

        # 1º perform a local aggregation in each partition
        partial_agg = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            partial_agg[f], info[f] = _aggregate(data[f], settings, f)

        # 2º perform a hash partition
        info = merge_reduce(merge_schema, info)

        from .repartition import hash_partition
        params = {'nfrag': nfrag,
                  'columns': settings['groupby'],
                  'info': [info]}

        repartition = hash_partition(partial_agg, params)['data']
        nfrag = len(repartition)

        # 3º perform a global aggregation
        info = [[] for _ in range(nfrag)]
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _merge_aggregation(repartition[f], settings, f)

        compss_delete_object(partial_agg)
        compss_delete_object(repartition)
        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output


@task(returns=2)
def _aggregate(data, params, f):
    """Perform a partial aggregation."""
    columns = params['groupby']
    target = params['aliases']
    operation = params['operation']

    if columns is '*':
        columns = list(data.columns)
        params['groupby'] = columns

    if '*' in target.keys():
        target[columns[0]] = target['*']
        del target['*']
        params['aliases'] = target

    if '*' in operation.keys():
        operation[columns[0]] = operation['*']
        del operation['*']
        params['operation'] = operation

    operation = _replace_functions_name(operation)
    data = data.groupby(columns).agg(operation)

    new_idx = []
    i = 0
    old = None
    # renaming
    for (n1, n2) in data.columns.ravel():
        if old != n1:
            old = n1
            i = 0
        new_idx.append(target[n1][i])
        i += 1

    data.columns = new_idx
    data = data.reset_index()
    data.reset_index(drop=True, inplace=True)

    info = generate_info(data, f)
    return data, info


@task(returns=2)
def _merge_aggregation(data1, params, f1):
    """Combining the aggregation with other fragment.

    if a key is present in both fragments, it will remain
    in the result only if f1 <f2.
    """
    columns = params['groupby']
    target = params['aliases']
    operation = params['operation']

    if len(data1) > 0:

        operation = _replace_name_by_functions(operation, target)
        data1 = data1.groupby(columns).agg(operation)

        # remove the different level
        data1.reset_index(inplace=True)
        sequence = columns + [c for c in data1.columns.tolist()
                              if c not in columns]
        data1 = data1[sequence]

    info = generate_info(data1, f1)
    return data1, info


def _collect_list(x):
    """Generate a list of a group."""
    return x.tolist()


def _collect_set(x):
    """Part of the generation of a set from a group.

    collect_list and collect_set must be different functions, otherwise
    pandas will raise error.
    """
    return x.tolist()


def _merge_set(series):
    """Merge set list."""
    return reduce(lambda x, y: list(set(x + y)), series.tolist())


def _replace_functions_name(operation):
    """Replace 'set' and 'list' to the pointer of the real function."""
    for col in operation.keys():
        for f in range(len(operation[col])):
            if operation[col][f] == 'list':
                operation[col][f] = _collect_list
            elif operation[col][f] == 'set':
                operation[col][f] = _collect_set
    return operation


def _replace_name_by_functions(operation, target):
    """Convert the operation dictionary to Alias."""
    new_operations = {}

    for col in operation:
        for f in range(len(operation[col])):
            if operation[col][f] == 'list':
                operation[col][f] = 'sum'
            elif operation[col][f] == 'set':
                operation[col][f] = _merge_set
            elif operation[col][f] == 'count':
                operation[col][f] = 'sum'

    for k in target:
        values = target[k]
        for i in range(len(values)):
            new_operations[values[i]] = operation[k][i]
    return new_operations
