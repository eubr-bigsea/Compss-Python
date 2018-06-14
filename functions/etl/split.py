#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.functions.reduce import mergeReduce
import numpy as np
import math


class SplitOperation(object):
    """Split Operation: Randomly splits a Data Frame into two dataframes."""

    def transform(self, data, settings, nfrag):
        """SplitOperation.

        :param data: A list with nfrag pandas's dataframe;
        :param settings: A dictionary that contains:
          - 'percentage': Percentage to split the data (default, 0.5);
          - 'seed': Optional, seed in case of deterministic random operation.
        :param nfrag: A number of fragments;
        :return: Returns two lists with nfrag pandas's dataframe with
                 distincts subsets of the input.

        Note: if percentage = 0.25, the final dataframes
              will have respectively, 25% and 75%.
        """

        idxs = self.preprocessing(settings, data, nfrag)
        splits1 = [[] for _ in range(nfrag)]
        splits2 = data[:]
        for i in range(nfrag):
            splits1[i] = _get_splits(splits2[i], idxs, i)

        return [splits1, splits2]

    def preprocessing(self, settings, data, nfrag):
        percentage = settings.get('percentage', 0.5)
        seed = settings.get('seed', None)

        if percentage < 0 or percentage > 1:
            raise Exception("Please inform a valid percentage [0, 1].")

        # count the size of each fragment and create a mapping
        # of the elements to be selected.
        partial_counts = [_count_record(d) for d in data]
        total = mergeReduce(_merge_count, partial_counts)
        idxs = _define_splits(total, percentage, seed, nfrag)
        return idxs


@task(returns=list, priority=True)
def _count_record(data):
    """Count the partial length."""
    size = len(data)
    return [size, [size]]


@task(returns=list, priority=True)
def _merge_count(df1, df2):
    """Merge the partial lengths."""
    return [df1[0]+df2[0], np.concatenate((df1[1], df2[1]), axis=0)]


# from pycompss.api.local import local
# or @local
@task(returns=list)
def _define_splits(total, percentage, seed, nfrag):
    """Define a list of indexes to be splitted."""
    total, n_list = total
    size = int(math.floor(total*percentage))

    np.random.seed(seed)
    ids = np.array(sorted(np.random.choice(total, size, replace=False)))

    n_list = np.cumsum(n_list)
    list_ids = [[] for _ in range(nfrag)]

    first_id = 0
    for i in range(nfrag):
        last_id = n_list[i]
        idx = (ids >= first_id) & (ids < last_id)
        list_ids[i] = ids[idx] - first_id
        first_id = last_id

    return list_ids


@task(data=INOUT, returns=list)
def _get_splits(data, indexes, frag):
    """Retrieve the split."""
    data.reset_index(drop=True, inplace=True)
    idx = data.index.isin(indexes[frag])
    split1 = data.loc[idx]
    data.drop(index=indexes[frag], inplace=True)
    return split1
