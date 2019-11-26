#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object
from ddf_library.utils import merge_schema, _get_schema, create_stage_files, \
    save_stage_file

import pandas as pd
import numpy as np
import sys


def parallelize(data, nfrag, stage_id):
    """
    Method to split the data in nfrag parts. This method simplifies
    the use of chunks.

    :param data: The np.array or list to do the split.
    :param nfrag: A number of partitions
    :return: A list of pandas DataFrame.

    :Note: the result may be unbalanced when the number of rows is too small
    """

    n_rows = len(data)

    # sizes = _generate_distribution1(n_rows, nfrag)  # only when n >> nfrag
    sizes = _generate_distribution2(n_rows, nfrag)

    cols = data.columns.tolist()
    data.reset_index(drop=True, inplace=True)
    result = create_stage_files(stage_id, nfrag)

    info = {'cols': cols,
            'dtypes': data.dtypes.values,
            'size': [],
            'memory': []
            }

    begin = 0
    for i, (n, out) in enumerate(zip(sizes, result)):
        partition = data.iloc[begin:begin+n]
        begin += n
        partition.reset_index(drop=True, inplace=True)
        save_stage_file(out, partition)
        info['size'].append(len(partition))
        info['memory'].append(sys.getsizeof(partition))

    if len(result) != nfrag:
        raise Exception("Error in parallelize function.")

    return result, info


def _generate_distribution1(n_rows, nfrag):
    """
    Each fragment will have the same number of rows, except
    in fragments at end and that doesnt have more available data.
    """
    size = n_rows / nfrag
    size = int(np.ceil(size))
    sizes = [size for _ in range(nfrag)]

    return sizes


def _generate_distribution2(n_rows, nfrag):
    """Data is divided among the partitions."""

    size = n_rows / nfrag
    size = int(np.ceil(size))
    sizes = [size for _ in range(nfrag)]

    i = 0
    while sum(sizes) > n_rows:
        i += 1
        sizes[i % nfrag] -= 1

    sizes = sorted(sizes, reverse=True)
    return sizes


def import_to_ddf(df_list, schema=None):
    """
    In order to import a list of DataFrames in DDF abstraction, we need to
    check the schema of each partition.

    :param df_list: a List of Pandas DataFrames
    :param schema: A list of columns names, data types and size in each fragment
    :return: a List of Pandas DataFrames and a schema
    """

    nfrag = len(df_list)

    if schema is None:
        schema = [_get_schema(df_list[f], f) for f in range(nfrag)]

    info_agg = merge_reduce(merge_schema, schema)
    compss_delete_object(schema)

    info_agg = compss_wait_on(info_agg)
    info = _check_schema(info_agg)

    return df_list, info


def _check_schema(info):

    if not isinstance(info, dict):
        raise Exception(info)

    memory = info['memory']
    human_readable = _human_bytes(sum(memory))
    avg = _human_bytes(np.mean(memory))
    median = _human_bytes(np.median(memory))

    print("Memory size of a "
          "partition: avg({}), median({})".format(avg, median))
    print("Total size of pandas in memory: ", human_readable)

    return info


def _human_bytes(size):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string"""
    power = 2**10
    n = 0
    power_of_n = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1

    value = "{:.2f} {}".format(size, power_of_n[n])
    return value
