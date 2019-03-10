#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task

import pandas as pd


class AggregationOperation(object):
    """Aggregation.

    Computes aggregates and returns the result as a DataFrame.
    """

    def transform(self, data, params):
        """AggregationOperation.

        :param data: A list with nfrag pandas's dataframe;
        :param params: A dictionary that contains:
            - groupby: A list with the columns names to aggregates;
            - alias: A dictionary with the aliases of all aggregated columns;
            - operation: A dictionary with the functions to be applied in
                         the aggregation:
                'mean': Computes average values for each numeric columns
                 for each group;
                'count': Counts the number of records for each group;
                'first': Returns the first element of group;
                'last': Returns the last element of group;
                'max': Computes the max value for each numeric columns for each group;
                'min': Computes the min value for each numeric column for each group;
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

        if 'aliases' not in params:
            params['aliases'] = {}
            for col in params['operation']:
                params['aliases'][col] = []
                ops = params['operation'][col]
                for op in ops:
                    alias = "{}({})".format(op, col)
                    params['aliases'][col].append(alias)

        nfrag = len(data)
        # the main ideia is to perform a local aggregation in each partition
        # and to compute the bounding box
        local_agg = [[] for _ in range(nfrag)]
        idx = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            local_agg[f], idx[f] = _aggregate(data[f], params)

        # them, group them if each pair is overlapping
        overlapping = overlap(idx)
        info = [[] for _ in range(nfrag)]
        result = local_agg[:]

        for f1 in range(nfrag):
            for f2 in range(nfrag):
                if f1 != f2 and overlapping[f1][f2]:
                    result[f1], info[f1] = _merge_aggregation(result[f1],
                                                              local_agg[f2],
                                                              params, f1, f2)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output


def overlap(sorted_idx):
    """Check if fragments A and B may have some elements to be joined."""
    nfrag = len(sorted_idx)
    overlapping = [[False for _ in range(nfrag)] for _ in range(nfrag)]
    from pycompss.api.api import compss_wait_on
    sorted_idx = compss_wait_on(sorted_idx)

    for i in range(nfrag):
        x_min, x_max = sorted_idx[i]

        if len(x_min) != 0:  # only if data1 was empty
            for j in range(nfrag):
                y_min, y_max = sorted_idx[j]

                if len(y_min) != 0:  # only if data2 was empty

                    tmp = pd.DataFrame([x_min, x_max, y_min, y_max],
                                       index=[0, 1, 2, 3])
                    tmp = tmp.infer_objects()
                    cols = [0]
                    tmp.sort_values(by=0, inplace=True)
                    idx = tmp.index

                    if any([idx[0] == 0 and idx[1] == 2,
                            idx[0] == 2 and idx[1] == 0,
                            all(tmp.iloc[0, cols] == tmp.iloc[2, cols])]):
                            overlapping[i][j] = True

    return overlapping


@task(returns=2)
def _aggregate(data, params):
    """Perform a partial aggregation."""
    columns = params['groupby']
    target = params['aliases']
    operation = params['operation']

    operation = _replace_functions_name(operation)
    data = data.groupby(columns).agg(operation)
    newidx = []
    i = 0
    old = None
    # renaming
    for (n1, n2) in data.columns.ravel():
        if old != n1:
            old = n1
            i = 0
        newidx.append(target[n1][i])
        i += 1

    data.columns = newidx
    data = data.reset_index()
    data.reset_index(drop=True, inplace=True)
    data.sort_values(columns, inplace=True)

    n = len(data)
    min_idx = data.loc[0, columns].values.tolist()
    max_idx = data.loc[n-1, columns].values.tolist()
    idx = [min_idx, max_idx]
    return data, idx


@task(returns=2)
def _merge_aggregation(data1, data2, params, f1, f2):
    """Combining the aggregation with other fragment.

    if a key is present in both fragments, it will remain
    in the result only if f1 <f2.
    """
    columns = params['groupby']
    target = params['aliases']
    operation = params['operation']

    if len(data1) > 0 and len(data2) > 0:

        # Keep only elements that is present in A
        merged = data2.merge(data1, on=columns,
                             how='left', indicator=True)
        data2 = data2.loc[merged['_merge'] != 'left_only', :]
        data2.reset_index(drop=True, inplace=True)

        # If f1>f2: Remove elements in data1 that is present in data2
        if f1 > f2:

            merged = data1.merge(data2, on=columns,
                                 how='left', indicator=True)
            data1 = data1.loc[merged['_merge'] != 'both', :]
            data1.reset_index(drop=True, inplace=True)
            if len(data2) > 0:
                merged = data2.merge(data1, on=columns,
                                     how='left', indicator=True)
                data2 = data2.loc[merged['_merge'] != 'left_only', :]

        operation = _replace_name_by_functions(operation, target)

        data1 = pd.concat([data1, data2], axis=0, ignore_index=True)
        data1 = data1.groupby(columns).agg(operation)

        # remove the different level
        data1.reset_index(inplace=True)

    info = [data1.columns.tolist(), data1.dtypes.values, [len(data1)]]
    return data1, info


def _collect_list(x):
    """Generate a list of a group."""
    return x.tolist()


def _collect_set(x):
    """Part of the generation of a set from a group.

    collect_list and collect_set must be diferent functions,
    otherwise pandas will raise error.
    """
    return x.tolist()


def _merge_set(series):
    """Merge set list."""
    return reduce(lambda x, y: list(set(x + y)), series.tolist())


def _replace_functions_name(operation):
    """Replace 'set' and 'list' to the pointer of the real function."""
    for col in operation:
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
