#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce

import numpy as np
import pandas as pd
import uuid
import sys

stage_id = 0


@task(returns=2)
def _generate_partition(size, f, dim, max_size):
    if max_size is None:
        max_size = size * 100

    cols = ["col{}".format(c) for c in range(dim)]
    df = pd.DataFrame({c: np.random.randint(0, max_size, size=size)
                       for c in cols})
    info = generate_info(df, f)
    return df, info


def generate_data(sizes, dim=1, max_size=None):

    nfrag = len(sizes)
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    for f, s in enumerate(sizes):
        dfs[f], info[f] = _generate_partition(s, f, dim, max_size)

    return dfs, info


def generate_info(df, f, info=None):
    if info is None:
        info = dict()
    info['cols'] = df.columns.tolist()
    info['dtypes'] = df.dtypes.values
    info['size'] = [len(df)]
    info['memory'] = [sys.getsizeof(df)]  # bytes
    info['frag'] = [f]

    return info


def merge_info(schemas):
    if isinstance(schemas, list):
        schemas = merge_reduce(merge_schema, schemas)
    return schemas


def concatenate_pandas(df):
    if any([True for r in df if len(r) > 0]):
        df = [r for r in df if len(r) > 0]  # to avoid change dtypes
    df = pd.concat(df, sort=False, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    return df


@task(returns=1)
def merge_schema(schema1, schema2):

    frags = np.array(schema1['frag'] + schema2['frag'])
    sizes = schema1['size'] + schema2['size']

    idx = np.argsort(frags)
    frags = frags[idx].tolist()
    sizes = np.array(sizes)[idx].tolist()

    schema = {'cols': schema1['cols'],
              'dtypes': schema1['dtypes'],
              'size': sizes,
              'frag': frags}

    if 'memory' in schema1:
        memory = schema1['memory'] + schema2['memory']
        schema['memory'] = np.array(memory)[idx].tolist()

    if set(schema1['cols']) != set(schema2['cols']):
        schema = "Error: Partitions have different columns names."

    return schema


@task(data_input=FILE_IN, returns=1)
def _get_schema(data_input, f):
    df = read_stage_file(data_input)
    info = generate_info(df, f)
    return info


def check_serialization(data):
    """
    Check if output is a Future file object or is data in-memory.
    :param data:
    :return:
    """

    if isinstance(data, list):
        if len(data) > 0:
            return isinstance(data[0], str)
        else:
            return False

    return isinstance(data, str)


def divide_idx_in_frags(ids, n_list):
    """
    Retrieve the real index (index in a fragment n) given a global index and
    the size of each fragment.

    :param ids:
    :param n_list:
    :return:
    """

    ids = np.sort(ids)

    list_ids = [[] for _ in n_list]
    top, bottom = 0, 0

    for frag, limit in enumerate(n_list):
        top += limit
        idx_bellow = [i for i in ids if i < top]
        list_ids[frag] = np.subtract(idx_bellow, bottom).tolist()
        bottom = top
        ids = ids[len(idx_bellow):]
        if len(ids) == 0:
            break

    return list_ids


def _gen_uuid():
    return str(uuid.uuid4())


def create_auxiliary_column(columns):
    condition = True
    column = "aux_column"
    while condition:
        column = _gen_uuid()[0:8]
        condition = column in columns
    return column


def col(name):
    from ddf_library.config import columns
    return columns.index(name)


def clear_context():
    from ddf_library.context import COMPSsContext, nx

    COMPSsContext.adj_tasks = dict()
    COMPSsContext.catalog = dict()
    COMPSsContext.tasks_map = dict()
    COMPSsContext.dag = nx.DiGraph()


def create_stage_files(nfrag, suffix=''):
    global stage_id
    file_names = ['/tmp/ddf_stage{}_{}block{}_{}.parquet'
                  .format(stage_id, suffix, f, _gen_uuid())
                  for f in range(nfrag)]
    stage_id += 1
    return file_names


def read_stage_file(filepath, cols=None):
    if isinstance(cols, str):
        cols = [cols]
    return pd.read_parquet(filepath, columns=cols)


def save_stage_file(filepath, df):
    return df.to_parquet(filepath)
