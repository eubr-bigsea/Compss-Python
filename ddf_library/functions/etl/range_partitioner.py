#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.parameter import FILE_IN, FILE_OUT, COLLECTION_IN
from pycompss.api.api import compss_wait_on, compss_delete_object

from ddf_library.utils import concatenate_pandas, generate_info, \
    create_stage_files, save_stage_file, read_stage_file

import pandas as pd
import numpy as np
import importlib


def range_partition(data, settings):
    """
    A Partitioner that partitions sortable records by range into roughly
    equal ranges. The ranges are determined by sampling the content of the
    DDF passed in.

    :param data: A list of pandas DataFrame;
    :param settings: A dictionary with:
     - info:
     - columns: Columns name to perform a range partition;
     - ascending:
     - nfrag: Number of partitions;

    :return: A list of pandas DataFrame;
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

    if nfrag_target < 0:
        raise Exception('You must have at least one partition.')

    # The actual number of partitions created by the RangePartitioner might
    # not be the same  as the nfrag parameter, in the case where the number of
    # sampled records is less than the value of nfrag.
    if sum(sizes) > 1:

        bounds, nfrag_target = range_bounds(data, nfrag, sizes, cols,
                                            ascending, nfrag_target)

        import ddf_library.config
        ddf_library.config.x = nfrag_target

        print("[INFO] - Number of partitions updated to: ", nfrag_target)

        splits = [[0 for _ in range(nfrag_target)] for _ in range(nfrag)]

        import ddf_library.functions.etl.repartition
        importlib.reload(ddf_library.functions.etl.repartition)

        for f in range(nfrag):
            splits[f] = ddf_library.functions.etl.repartition\
                .split_by_boundary(data[f], cols, ascending, bounds,
                                   info, nfrag_target)

        result = create_stage_files(nfrag_target)
        info = [{} for _ in range(nfrag_target)]
        for f in range(nfrag_target):
            if nfrag_target == 1:
                tmp = splits
            else:
                tmp = [splits[t][f] for t in range(nfrag)]
            tmp = [splits[t][f] for t in range(nfrag)]
            info[f] = concat_n_pandas(result[f], f, tmp)

        compss_delete_object(splits)

    else:
        result = data

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def range_bounds(data, nfrag, sizes, cols, ascending, nfrag_target):
    n_rows = sum(sizes)

    total_sample_size = min([60 * nfrag, 1e6])
    if nfrag >= n_rows:
        total_sample_size = n_rows - 2 if n_rows - 2 > 0 else 1
        nfrag_target = n_rows - 2 if n_rows - 2 > 0 else 2

    ratio = total_sample_size / n_rows
    sample_size = [int(np.ceil(ratio * s)) for s in sizes]

    sample_idx = [_sample_keys(data[f], cols,
                               sample_size[f]) for f in range(nfrag)]
    sample_idx = compss_wait_on(sample_idx)
    sample_idx = concatenate_pandas(sample_idx)

    bounds = _determine_bounds(sample_idx, cols, ascending, nfrag_target)
    nfrag_target = len(bounds) + 1

    return bounds, nfrag_target


@task(returns=1, input_data=FILE_IN)
def _sample_keys(input_data, cols, sample_size):
    data = read_stage_file(input_data, cols)
    data = data[cols] # TODO
    n = len(data)
    sample_size = sample_size if sample_size < n else n
    data.reset_index(drop=True, inplace=True)
    data = data.sample(n=sample_size, replace=False)
    return data


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


# @constraint(ComputingUnits="2")  # approach to have more memory
@task(data_out=FILE_OUT, args=COLLECTION_IN, returns=1)
def concat_n_pandas(data_out, f, args):
    dfs = [df for df in args if isinstance(df, pd.DataFrame)]
    dfs = pd.concat(dfs, ignore_index=True, sort=False)
    del args
    dfs = dfs.infer_objects()
    info = generate_info(dfs, f)
    save_stage_file(data_out, dfs)
    return info
