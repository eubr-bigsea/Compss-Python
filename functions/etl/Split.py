#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Split Operation: Randomly splits a Data Frame into two dataframes."""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
import numpy as np
import math


class SplitOperation(object):

    def __init__(self):
        pass

    def transform(self, data, settings, numFrag):
        """SplitOperation.

        :param data: A list with numFrag pandas's dataframe;
        :settings: A dictionary that contains:
          - 'percentage': Percentage to split the data (default, 0.5);
          - 'seed': Optional, seed in case of deterministic random operation.
        :return: Returns two lists with numFrag pandas's dataframe with
                 distincts subsets of the input.

        Note: if percentage = 0.25, the final dataframes
              will have respectively, 25% and 75%.
        """
        percentage = settings.get('percentage', 0.5)
        seed = settings.get('seed', None)

        if percentage < 0 or percentage > 1:
            raise Exception("Please inform a valid percentage [0, 1].")

        # count the size of each fragment and create a mapping
        # of the elements to be selected.
        partial_counts = [self._count_record(data[i]) for i in range(numFrag)]
        total = mergeReduce(self.mergeCount, partial_counts)
        indexes = self.define_splits(total, percentage, seed, numFrag)

        splits1 = [[] for i in range(numFrag)]
        splits2 = [[] for i in range(numFrag)]
        for i in range(numFrag):
            splits1[i] = self.get_splits(data[i], indexes, True, i)
            splits2[i] = self.get_splits(data[i], indexes, False, i)

        return [splits1, splits2]


    @task(returns=list)
    def _count_record(self, data):
        """Count the partial length."""
        size = len(data)
        return [size, [size]]


    @task(returns=list)
    def mergeCount(self, df1, df2):
        """Merge the partial lengths."""
        return [df1[0]+df2[0], np.concatenate((df1[1], df2[1]), axis=0)]


    @task(returns=list)
    def define_splits(self, N_list, percentage, seed, numFrag):
        """Define a list of indexes to be splitted."""
        total, n_list = N_list
        size = int(math.floor(total*percentage))

        np.random.seed(seed)
        ids = np.array(sorted(np.random.choice(total, size, replace=False)))

        n_list = np.cumsum(n_list)
        list_ids = [[] for i in range(numFrag)]

        first_id = 0
        for i in range(numFrag):
            last_id = n_list[i]
            idx = (ids >= first_id) & (ids < last_id)
            list_ids[i] = ids[idx]
            first_id = last_id

        return list_ids


    @task(returns=list)
    def get_splits(self, data, indexes, part1, frag):
        """Retrieve the split."""
        data = data.reset_index(drop=True)

        if part1:
            split = data.loc[data.index.isin(indexes[frag])]
        else:
            split = data.loc[~data.index.isin(indexes[frag])]

        return split
