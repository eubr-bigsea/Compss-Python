#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task


class CrossJoinOperation(object):

    def transform(self, data1, data2):
        """
        Returns the cartesian product with another DataFrame.

        :param data1: A list with nfrag pandas's dataframe;
        :param data2: A list with nfrag pandas's dataframe;
        :return: Returns a list with nfrag pandas's dataframe.
        """

        if isinstance(data[0], pd.DataFrame):
            result = copy.deepcopy(data)
        else:
            # when using deepcopy and variable is FutureObject
            # list, COMPSs is not able to restore in worker
            result = data[:]

        x_i, y_i = self.preprocessing(nfrag)
        for x, y in zip(x_i, y_i):
            result[x], result[y] = _drop_duplicates(result[x], result[y], cols)

        return result

    def preprocessing(self, nfrag):
        import itertools
        buff = list(itertools.combinations([x for x in range(nfrag)], 2))

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
