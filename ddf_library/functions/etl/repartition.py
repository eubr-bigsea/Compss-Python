#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.constraint import constraint

from ddf_library.utils import concatenate_pandas, generate_info, \
    create_auxiliary_column
from .parallelize import _generate_distribution2
from .balancer import _balancer

import numpy as np
import pandas as pd
import xxhash
import time


def repartition(data, settings):
    """

    :param data: a list of pandas dataframes;
    :param settings: A dictionary with:
        - 'shape': A list with the number of rows in each fragment.
        - 'nfrag': The new number of fragments.
    :return: A list of pandas dataframes;

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


def range_partition(data, settings):
    """
    A Partitioner that partitions sortable records by range into roughly
    equal ranges. The ranges are determined by sampling the content of the
    DDF passed in.

    :param data: A list of pandas dataframes;
    :param settings: A dictionary with:
     - info:
     - columns: Columns name to perform a range partition;
     - ascending:
     - nfrag: Number of partitions;

    :return: A list of pandas dataframes;
    """

    info = settings['info'][0]
    sizes = info['size']
    nfrag = len(data)
    cols = settings['columns']
    nfrag_target = settings.get('nfrag', nfrag)
    ascending = settings.get('ascending', True)

    if not isinstance(cols, list):
        cols = [cols]

    if not isinstance(ascending, list):
        ascending = [ascending] * len(cols)

    if nfrag_target < 1:
        raise Exception('You must have at least one partition.')

    # The actual number of partitions created by the RangePartitioner might
    # not be the same  as the nfrag parameter, in the case where the number of
    # sampled records is less than the value of nfrag.
    if sum(sizes) > 1:

        bounds, nfrag_target = range_bounds(data, nfrag, sizes, cols,
                                            ascending, nfrag_target)

        print("[INFO] - Number of partitions update to: ", nfrag_target)

        splitted = [0 for _ in range(nfrag)]
        # create a matrix 2-D where:
        # row is each original frag in splitted, nfrag
        # column is each bucket in splitted, next divisible of nfrag_target
        nfrags_new = find_next_divisible(nfrag_target, 10)
        frags = [[[] for _ in range(nfrags_new)] for _ in range(nfrag)]

        for f in range(nfrag):
            splitted[f] = split_by_boundary(data[f], cols, ascending,
                                            bounds, info, nfrags_new)

            for f2 in range(0, nfrags_new, 10):
                frags[f][f2:f2+10] = get_partition(splitted[f], f2, f2+10)
        compss_delete_object(splitted)

        result = [[] for _ in range(nfrag_target)]
        info = [{} for _ in range(nfrag_target)]

        for f in range(nfrag_target):
            fs = [frags[t][f] for t in range(nfrag)]
            tmp = merge_reduce(concat2pandas, fs[0:-1])
            result[f], info[f] = _gen_partition(tmp, fs[-1], f)

        compss_delete_object(frags)

    else:
        result = data

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def find_next_divisible(m, n):
    return n * int(np.ceil(m / n))


def range_bounds(data, nfrag, sizes, cols, ascending, nfrag_target):
    n_rows = sum(sizes)

    total_sample_size = min([60 * nfrag, 1e6])
    if nfrag >= n_rows:
        total_sample_size = n_rows - 2 if n_rows - 2 > 0 else 1
        nfrag_target = n_rows - 2 if n_rows - 2 > 0 else 2

    ratio = total_sample_size / n_rows
    sample_size = [int(np.ceil(ratio * s)) for s in sizes]

    sample_idxs = [_sample_keys(data[f], cols, sample_size[f])
                   for f in range(nfrag)]
    sample_idxs = compss_wait_on(sample_idxs)
    sample_idxs = concatenate_pandas(sample_idxs)

    bounds = _determine_bounds(sample_idxs, cols, ascending, nfrag_target)
    nfrag_target = len(bounds) + 1

    return bounds, nfrag_target


def _determine_bounds(sample, cols, ascending, nfrag_target):

    sample = sample.groupby(cols).size().reset_index(name='counts')
    sample.sort_values(by=cols, ascending=ascending, inplace=True)
    sample = sample.values

    step = np.sum(sample[:, -1]) / nfrag_target
    num_candidates = len(sample)

    key = len(cols)
    cum_weight = 0.0
    i, j = 0, 0
    target = step

    bounds = []
    while (i < num_candidates) and (j < nfrag_target - 1):

        weight = sample[i][-1]
        cum_weight += weight

        if cum_weight >= target:
            bounds.append(sample[i][0:key])
            target += step
            j += 1
        i += 1

    return bounds


@constraint(ComputingUnits="2")  # approach to have more available memory
@task(returns=1)
def split_by_boundary(data, cols, ascending, bounds, info, nfrags_new):
    # doesnt need to properly sort, other approach like nargsort might be work

    splits = [pd.DataFrame(columns=info['cols'], dtype=info['dtypes'])
              for _ in range(nfrags_new)]
    aux_col = create_auxiliary_column(info['cols'])

    # creation of keys DataFrame with all bounds
    bounds = pd.DataFrame(bounds, columns=cols)
    bounds[aux_col] = -1
    keys = data[cols]
    keys[aux_col] = data.index
    keys = pd.concat([keys, bounds], sort=False)
    del bounds

    t1 = time.time()
    # sort only the DataFrame with columns. It consumes more space,
    # but is faster
    keys.sort_values(by=cols, ascending=ascending, inplace=True)
    keys.reset_index(drop=True, inplace=True)
    t2 = time.time()
    print("sort in split_by_boundary: ", t2-t1)

    idxs_bounds = keys.index[keys[aux_col] == -1]
    aux_col_idx = keys.columns.get_loc(aux_col)

    for s, idx in enumerate(idxs_bounds):

        if s == 0:
            list_idx = keys.iloc[0:idx, aux_col_idx]
            splits[s] = data.iloc[list_idx]

        else:
            idx0 = idxs_bounds[s-1]
            list_idx = keys.iloc[idx0 + 1:idx, aux_col_idx]
            splits[s] = data.iloc[list_idx]

        if (s+1) == len(idxs_bounds):
            list_idx = keys.iloc[idx + 1:, aux_col_idx]
            splits[s+1] = data.iloc[list_idx]

    return splits


@task(returns=1)
def _sample_keys(data, cols, sample_size):
    data = data[cols]
    n = len(data)
    sample_size = sample_size if sample_size < n else n
    data.reset_index(drop=True, inplace=True)
    data = data.sample(n=sample_size, replace=False)
    return data


@task(returns=10)
def get_partition(splits, fin, fout):
    return splits[fin: fout]


@constraint(ComputingUnits="2")  # approach to have more available memory
@task(returns=1)
def concat2pandas(df1, df2):
    return pd.concat([df1, df2], ignore_index=True, sort=False)


@task(returns=2)
def _gen_partition(df, df2, frag):

    df = pd.concat([df, df2], ignore_index=True, sort=False)
    df.reset_index(drop=True, inplace=True)
    info = generate_info(df, frag)
    return df, info


def hash_partition(data, settings):
    """
    A Partitioner that partitions are organized by hash function.

    :param data: A list of pandas dataframes;
    :param settings: A dictionary with:
     - info:
     - columns: Columns name to perform a hash partition;
     - nfrag: Number of partitions;

    :return: A list of pandas dataframes;
    """
    info = settings['info'][0]
    nfrag = len(data)
    cols = settings['columns']
    nfrag_target = settings.get('nfrag', nfrag)

    if nfrag_target < 1:
        raise Exception('You must have at least one partition.')

    elif nfrag_target > 1:

        splitted = [split_by_hash(data[f], cols, info, nfrag_target)
                    for f in range(nfrag)]

        result = [[] for _ in range(nfrag_target)]
        info = [{} for _ in range(nfrag_target)]

        for f in range(nfrag_target):
            for f2 in range(nfrag):
                result[f], info[f] = merge_splits(result[f], splitted[f2], f)

    else:
        result = data

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@task(returns=1)
def split_by_hash(df, cols, info, nfrag):
    splits = {f: pd.DataFrame(columns=info['cols'], dtype=info['dtypes'])
              for f in range(nfrag)}

    df.reset_index(drop=True, inplace=True)
    if len(df) > 0:

        keys = df[cols].astype(str).values.sum(axis=1).reshape((-1, 1))
        keys.flags.writeable = False
        idxs = np.apply_along_axis(hashcode, 1, keys) % nfrag

        for f in range(nfrag):
            idx = np.argwhere(idxs == f).flatten()
            if len(idx) > 0:
                splits[f] = df.iloc[idx]

    return splits


def hashcode(x):
    return xxhash.xxh64_intdigest(x[0], seed=42)
    # option 2:
    # return hash(x)
    # option 3:
    # from pyhashxx import Hashxx
    # return = Hashxx(x[0], seed=42).digest()


@task(returns=2)
def merge_splits(df1, df2, frag):

    df2 = df2[frag]

    if len(df1) == 0:
        df1 = df2
    elif len(df2) > 0:
        df1 = pd.concat([df1, df2], ignore_index=True, sort=False)

    df1.reset_index(drop=True, inplace=True)

    info = generate_info(df1, frag)

    return df1, info


"""

para dividir... basta juntar algumas particoes

"""
# def coalesce(data, settings):
#     """
#
#     :param data:
#     :param settings: A dictionary with:
#         - 'nfrag': The new number of fragments.
#     :return:
#
#     .. note: coalesce uses existing partitions to minimize the amount of data
#     that's shuffled.  repartition creates new partitions and does a full
#     shuffle. coalesce results in partitions with different amounts of data
#     (sometimes partitions that have much different sizes) and repartition
#     results in roughly equal sized partitions.
#     """
#
#     info = settings['info'][0]
#     target_dist = settings.get('shape', [])
#     nfrag = settings.get('nfrag', len(data))
#     if nfrag < 1:
#         nfrag = len(data)
#
#     old_sizes = info['size']
#     cols = info['cols']
#
#     if len(target_dist) == 0:
#         n_rows = sum(old_sizes)
#         target_dist = _generate_distribution2(n_rows, nfrag)
#
#     result, info = _balancer(data, target_dist, old_sizes, cols)
#
#     output = {'key_data': ['data'], 'key_info': ['info'],
#               'data': result, 'info': info}
#     return output
