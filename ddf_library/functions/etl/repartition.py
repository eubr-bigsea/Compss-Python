#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task

from pycompss.api.constraint import constraint

from ddf_library.utils import create_auxiliary_column
from .parallelize import _generate_distribution2
from .balancer import _balancer
import ddf_library.config as config

import numpy as np
import pandas as pd
import zlib
import time


def repartition(data, settings):
    """

    :param data: a list of pandas DataFrames;
    :param settings: A dictionary with:
        - 'shape': A list with the number of rows in each fragment.
        - 'nfrag': The new number of fragments.
    :return: A list of pandas DataFrames;

    .. note: 'shape' has prevalence over 'nfrag'

    .. note: coalesce uses existing partitions to minimize the amount of data
    that's shuffled.  repartition creates new partitions and does a full
    shuffle. coalesce results in partitions with different amounts of data
    (sometimes partitions that have much different sizes) and repartition
    results in roughly equal sized partitions.
    """

    info = settings['info'][0]
    target_dist = settings.get('shape', [])
    nfrag = settings.get('nfrag', len(data))

    if nfrag < 1:
        nfrag = len(data)

    old_sizes = info['size']
    cols = info['cols']

    if len(target_dist) == 0:
        n_rows = sum(old_sizes)
        target_dist = _generate_distribution2(n_rows, nfrag)

    result, info = _balancer(data, target_dist, old_sizes, cols)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@task(returns=config.x)
def split_by_hash(df, cols, info, nfrag):
    splits = [pd.DataFrame(columns=info['cols'], dtype=info['dtypes'])
              for _ in range(nfrag)]

    df.reset_index(drop=True, inplace=True)
    if len(df) > 0:

        keys = df[cols].astype(str).values.sum(axis=1).ravel()
        keys.flags.writeable = False
        v_hashcode = np.vectorize(hashcode)
        indexes = v_hashcode(keys) % nfrag

        for f in range(nfrag):
            idx = np.argwhere(indexes == f).flatten()
            if len(idx) > 0:
                splits[f] = df.iloc[idx]

    return splits


def hashcode(x):
    return int(zlib.adler32(x.encode()) & 0xffffffff)


@constraint(ComputingUnits="2")  # approach to have more available memory
@task(returns=config.x)
def split_by_boundary(data, cols, ascending, bounds, info, nfrag):
    splits = [pd.DataFrame(columns=info['cols'], dtype=info['dtypes'])
              for _ in range(nfrag)]

    if len(data) > 0:
        aux_col = create_auxiliary_column(info['cols'])
        data.reset_index(drop=True, inplace=True)

        # creation of keys DataFrame with all bounds
        bounds = pd.DataFrame(bounds, columns=cols)
        bounds[aux_col] = -1
        keys = data[cols].copy()
        keys[aux_col] = keys.index
        keys = pd.concat([keys, bounds], sort=False)
        del bounds

        t1 = time.time()
        # sort only the DataFrame with columns. It consumes more space,
        # but is faster
        keys.sort_values(by=cols, ascending=ascending, inplace=True)
        keys = keys[[aux_col]]
        keys.reset_index(drop=True, inplace=True)
        t2 = time.time()
        print("sort in split_by_boundary: ", t2-t1)

        idx_bounds = keys.index[keys[aux_col] == -1]
        aux_col_idx = keys.columns.get_loc(aux_col)

        for s, idx in enumerate(idx_bounds):

            if s == 0:
                list_idx = keys.iloc[0:idx, aux_col_idx].values
            else:
                idx0 = idx_bounds[s-1]
                list_idx = keys.iloc[idx0 + 1:idx, aux_col_idx].values

            # list_idx = np.sort(list_idx)
            splits[s] = data.iloc[list_idx]

            if (s+1) == len(idx_bounds):
                list_idx = keys.iloc[idx + 1:, aux_col_idx].values
                # list_idx = np.sort(list_idx)
                splits[s+1] = data.iloc[list_idx]

    if nfrag == 1:
        splits = splits[0]

    return splits


def merge_n_reduce(f, data, n):
    """
    Apply f cumulatively to the items of data,
    from left to right in binary tree structure, so as to
    reduce the data to a single value.

    :param f: function to apply to reduce data
    :param data: List of items to be reduced
    :param n: step size
    :return: result of reduce the data to a single value
    """

    from collections import deque
    q = deque(range(len(data)))
    new_data = data[:]
    len_q = len(q)
    while len_q:
        x = q.popleft()
        len_q = len(q)
        if len_q:
            min_d = min([len_q, n-1])
            xs = [q.popleft() for _ in range(min_d)]
            xs = [new_data[i] for i in xs] + [0] * (n - min_d - 1)

            new_data[x] = f(new_data[x], *xs)
            q.append(x)

        else:
            return new_data[x]
