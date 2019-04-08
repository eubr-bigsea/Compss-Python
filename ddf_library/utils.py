#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
import numpy as np
import pandas as pd
import sys


@task(returns=2)
def _generate_partition(size, f):
    df = pd.DataFrame({'a': np.random.randint(0, size * 100, size=size)})
    info = generate_info(df, f)
    return df, info


def generate_data(sizes):

    nfrag = len(sizes)
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    for f, s in enumerate(sizes):
        dfs[f], info[f] = _generate_partition(s, f)

    return dfs, info


def generate_info(df, f):
    info = {'cols': df.columns.tolist(),
            'dtypes': df.dtypes.values,
            'size': [len(df)],
            'memory': [sys.getsizeof(df)],  # bytes
            'frag': [f]
            }
    return info


def merge_info(schemas):
    return merge_reduce(merge_schema, schemas)


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

    # dtypes = info['dtypes'][0]  # TODO

    return schema
