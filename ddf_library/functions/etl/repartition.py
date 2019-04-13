#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object

from .parallelize import _generate_distribution2
from .balancer import _balancer
from ddf_library.utils import concatenate_pandas, generate_info
import numpy as np
import pandas as pd


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
     - size:
     - columns:
     - ascending:
     - nfrag:

    :return: A list of pandas dataframes;
    """

    info = settings['info'][0]
    sizes = info['size']
    nfrag = len(data)
    cols = settings['columns']
    nfrag_target = settings.get('nfrag', nfrag)
    ascending = settings.get('ascending', True)

    if not isinstance(ascending, list):
        ascending = [ascending for _ in cols]

    if nfrag_target < 1:
        raise Exception('You must have at least one partition.')

    # The actual number of partitions created by the RangePartitioner might
    # not be the same  as the nfrag parameter, in the case where the number of
    # sampled records is less than the value of nfrag.
    if sum(sizes) > 1:

        bounds, nfrag_target = range_bounds(data, nfrag, sizes, cols,
                                            ascending, nfrag_target)

        print("[INFO] - Number of partitions update to: ", nfrag_target)

        splitted = [split_by_boundary(data[f], cols, ascending,
                                      bounds, info) for f in range(nfrag)]

        result = [[] for _ in range(nfrag_target)]
        info = [{} for _ in range(nfrag_target)]

        # for f2 in range(nfrag):
        #     for f in range(nfrag_target):
        #         result[f] = merge_partitions(result[f], splitted[f2],
        #         f, info[f])
        #     compss_delete_object(splitted[f2])

        for f in range(nfrag_target):
            frags = [get_partition(splitted[f2], f) for f2 in range(nfrag)]
            tmp = merge_reduce(concat2pandas, frags)
            compss_delete_object(frags)
            result[f], info[f] = _gen_partition(tmp, f)
            compss_delete_object(tmp)

    else:
        result = data

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


# @task(returns=1, info=INOUT)
# def merge_partitions(df, df2, frag, info):
#     tmp = df2[frag]
#     del df2
#
#     if len(df) == 0:
#         df = tmp
#     else:
#         df = pd.concat([df, tmp], ignore_index=True)
#
#     generate_info(df, frag, info)
#     return df


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


@task(returns=1)
def split_by_boundary(data, cols, ascending, bounds, info):
    # TODO: find a better way to select rows between ranges

    n_splits = len(bounds)+1
    splits = {f: pd.DataFrame(columns=info['cols'], dtype=info['dtypes'])
              for f in range(n_splits)}

    aux_col = 'aux_split_by_boundary'
    bounds = pd.DataFrame(bounds, columns=cols)
    bounds[aux_col] = -1

    tmp = data[cols].copy()
    tmp[aux_col] = tmp.index

    values_bounds = pd.concat([tmp, bounds])
    del tmp, bounds
    values_bounds.sort_values(by=cols, ascending=ascending, inplace=True)
    values_bounds.reset_index(drop=True, inplace=True)

    idxs_bounds = values_bounds.index[values_bounds[aux_col] == -1]

    for s, idx in enumerate(idxs_bounds):

        if s == 0:
            t = values_bounds.iloc[0:idx]
            t.drop(aux_col, axis=1, inplace=True)
            splits[s] = t

        else:
            idx0 = idxs_bounds[s-1]
            t = values_bounds.iloc[idx0 + 1:idx]
            t.drop(aux_col, axis=1, inplace=True)
            splits[s] = t

        if (s+1) == len(idxs_bounds):
            t = values_bounds.iloc[idx + 1:]
            t.drop(aux_col, axis=1, inplace=True)
            splits[s+1] = t

    return splits


@task(returns=1)
def get_partition(splits, frag):
    return [splits[frag]]


@task(returns=1)
def concat2pandas(df1, df2):
    return df1 + df2
    # return pd.concat([df1, df2], ignore_index=True)


@task(returns=2)
def _gen_partition(df, frag):

    if len(df) > 1:
        df = pd.concat(df, ignore_index=True, sort=False)
    else:
        df = df[0]

    info = generate_info(df, frag)

    return df, info


@task(returns=1)
def _sample_keys(data, cols, sample_size):
    data = data[cols]
    n = len(data)
    sample_size = sample_size if sample_size < n else n
    data.reset_index(drop=True, inplace=True)
    data = data.sample(n=sample_size, replace=False)
    return data


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
