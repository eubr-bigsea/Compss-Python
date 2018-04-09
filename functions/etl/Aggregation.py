#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Aggregation: Computes aggregates and returns the result as a DataFrame."""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
import pandas as pd


class AggregationOperation(object):

    def __init__(self):
        pass

    def transform(self, data, params, numFrag):
        """AggregationOperation.

        :param data: A list with numFrag pandas's dataframe;
        :param params: A dictionary that contains:
            - columns: A list with the columns names to aggregates;
            - alias: A dictionary with the aliases of all aggregated columns;
            - operation: A dictionary with the functionst to be applied in
                         the aggregation:
                'mean': Computes the average of each group;
                'count': Counts the total of records of each group;
                'first': Returns the first element of group;
                'last': Returns the last element of group;
                'max': Returns the max value of each group for one attribute;
                'min': Returns the min value of each group for one attribute;
                'sum': Returns the sum of values of each group for one
                       attribute;
                'list': Returns a list of objects with duplicates;
                'set': Returns a set of objects with duplicate elements
                            eliminated.
        :param numFrag: The number of fragments;
        :return: Returns a list with numFrag pandas's dataframe.

        example:
            settings['columns']   = ["col1"]
            settings['operation'] = {'col2':['sum'],'col3':['first','last']}
            settings['aliases']   = {'col2':["Sum_col2"],
                                     'col3':['col_First','col_Last']
                                    }
        """

        tmp = [[] for f in range(numFrag)]
        result = [tmp[f] for f in range(numFrag)]
        for f in range(numFrag):
            tmp[f] = self._aggregate(data[f], params)

        # passar pra baixo

        for f1 in range(numFrag):
            for f2 in range(numFrag):
                if f1 != f2:
                    result[f1] = self.merge_aggregation(result[f1], tmp[f2],
                                                 params, f1, f2)

        return result


    def create_execution_list(self, numFrag):
        """Create a list of execution."""
        # buffer to store the join between each block
        import itertools
        buff = list(itertools.combinations([x for x in range(numFrag)], 2))

        # Merging the partial results
        def disjoint(a, b):
            return set(a).isdisjoint(b)

        x_i = []
        y_i = []

        while len(buff) > 0:
            x = buff[0][0]
            step_list_i = []
            step_list_j = []
            if x >= 0:
                y = buff[0][1]
                step_list_i.append(x)
                step_list_j.append(y)
                buff[0] = [-1, -1]
                for j in range(len(buff)):
                    tuples = buff[j]
                    if tuples[0] >= 0:
                        if disjoint(tuples, step_list_i):
                            if disjoint(tuples, step_list_j):
                                step_list_i.append(tuples[0])
                                step_list_j.append(tuples[1])
                                buff[j] = [-1, -1]
            del buff[0]
            x_i.extend(step_list_i)
            y_i.extend(step_list_j)
        return x_i, y_i


    @task(returns=list)
    def _aggregate(self, data, params):
        """Perform a partial aggregation."""
        columns = params['columns']
        target = params['aliases']
        operation = params['operation']
        operation = self.replace_functions_name(operation)
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
        data = data.reset_index(drop=True)
        return data


    @task(returns=list)
    def merge_aggregation(self, data1, data2, params, f1, f2):
        """Combining the aggregation with other fragment.

        if a key is present in both fragments, it will remain
        in the result only if f1 <f2.
        """
        columns = params['columns']
        target = params['aliases']
        operation = params['operation']

        if len(data1) > 0 and len(data2) > 0:
            # Keep only elements that is present in A
            merged = data2.merge(data1, on=columns,
                                 how='left', indicator=True)
            data2 = data2.loc[merged['_merge'] != 'left_only', :]
            data2 = data2.reset_index(drop=True)

            # If f1>f2: Remove elements in data1 that is present in data2
            if f1 > f2:

                merged = data1.merge(data2, on=columns,
                                     how='left', indicator=True)
                data1 = data1.loc[merged['_merge'] != 'both', :]
                data1 = data1.reset_index(drop=True)
                if len(data2) > 0:
                    merged = data2.merge(data1, on=columns,
                                         how='left', indicator=True)
                    data2 = data2.loc[merged['_merge'] != 'left_only', :]

            operation = self.replace_name_by_functions(operation, target)

            data = pd.concat([data1, data2], axis=0, ignore_index=True)
            data = data.groupby(columns).agg(operation)

            # remove the diffent level
            data.reset_index(inplace=True)
            return data
        return data1


    def collectList(self, x):
        """Generate a list of a group."""
        return x.tolist()


    def collectSet(self, x):
        """Part of the generation of a set from a group.

        CollectList and collectSet must be diferent functions,
        otherwise pandas will raise error.
        """
        return x.tolist()


    def mergeSet(self, series):
        """Merge set list."""
        return reduce(lambda x, y: list(set(x + y)), series.tolist())


    def replace_functions_name(self, operation):
        """Replace 'set' and 'list' to the pointer of the real function."""
        for col in operation:
            for f in range(len(operation[col])):
                if operation[col][f] == 'list':
                    operation[col][f] = self.collectList
                elif operation[col][f] == 'set':
                    operation[col][f] = self.collectSet
        return operation


    def replace_name_by_functions(self, operation, target):
        """Convert the operation dictionary to Alias."""
        new_operations = {}

        for col in operation:
            for f in range(len(operation[col])):
                if operation[col][f] == 'list':
                    operation[col][f] = 'sum'
                elif operation[col][f] == 'set':
                    operation[col][f] = self.mergeSet
                elif operation[col][f] == 'count':
                    operation[col][f] = 'sum'

        for k in target:
            values = target[k]
            for i in range(len(values)):
                new_operations[values[i]] = operation[k][i]
        return new_operations
