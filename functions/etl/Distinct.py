#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.parameter import *
from pycompss.api.task import task
import pandas as pd


class DistinctOperation(object):
    """Distinct Operation: Remove duplicates elements."""

    def transform(self, data, cols, numFrag):
        """DistinctOperation.

        :param data: A list with numFrag pandas's dataframe;
        :param cols: A list with the columns names to take in count
                    (if no field is choosen, all fields are used).
        :param numFrag: The number of fragments;
        :return: Returns a list with numFrag pandas's dataframe.
        """
        result = data[:]
        import itertools
        buff = list(itertools.combinations([x for x in range(numFrag)], 2))

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

        for x, y in zip(x_i, y_i):
            self._drop_duplicates(result[x], result[y], cols)

        return result

    @task(isModifier=False, data1=INOUT, data2=INOUT)
    def _drop_duplicates(self, data1, data2, cols):
        """Remove duplicate rows based in two fragments at the time."""
        data = pd.concat([data1, data2], axis=0, ignore_index=True)
        n = len(data)
        if n > 0:
            alls_cols = data1.columns
            n1 = len(data1)
            index = ['p1' for x in range(n1)] + ['p2' for x in xrange(n1, n)]
            data['dropDup_index'] = index

            # if no field is choosen, all fields are used)
            if len(cols) == 0:
                cols = data.columns

            data = data.drop_duplicates(cols, keep='first')\
                       .reset_index(drop=True)
            data1.reset_index(drop=True, inplace=True)
            data2.reset_index(drop=True, inplace=True)
            # se os dados tiverem exatamente o mesmo tipo, pode ser feito
            # algo mais eficiente. Por exemplo, usando infer_objects()
            tmp1 = data.loc[data['dropDup_index'] == 'p1', alls_cols].values
            m1 = len(tmp1)
            data1.iloc[0:m1, :] = tmp1
            tmp1 = data.loc[data['dropDup_index'] == 'p2', alls_cols].values
            m2 = len(tmp1)
            data2.iloc[0:m2, :] = tmp1

            data1.drop(data1.index[m1:], inplace=True)
            data2.drop(data2.index[m2:], inplace=True)
