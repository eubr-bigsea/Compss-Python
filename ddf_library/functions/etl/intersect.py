#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from .hash_partitioner import hash_partition

import pandas as pd


def intersect_stage_1(data1, data2, settings):

    info1, info2 = settings['schema']
    nfrag = len(data1)

    params_hash1 = {'columns': [], 'schema': [info1], 'nfrag': nfrag}
    params_hash2 = {'columns': [], 'schema': [info2], 'nfrag': nfrag}

    out1 = hash_partition(data1, params_hash1)
    out2 = hash_partition(data2, params_hash2)

    data1 = out1['data']
    data2 = out2['data']

    return data1, data2, settings


def intersect_stage_2(df1, df2, settings):
    """Perform a partial intersection."""

    remove_duplicates = settings.get('distinct', True)
    frag = settings['id_frag']

    keys = df1.columns.tolist()
    keys2 = df2.columns.tolist()

    df1 = df1.dropna(axis=0, how='any')

    if remove_duplicates:
        df1 = df1.drop_duplicates(subset=keys, ignore_index=True)

    df2 = df2.dropna(axis=0, how='any')\
        .drop_duplicates(subset=keys2, ignore_index=True)

    if set(keys) == set(keys2) and len(df1) > 0:
        df1 = pd.merge(df1, df2, how='inner', on=keys, copy=False)
        df1 = df1.infer_objects()

    info = generate_info(df1, frag)
    return df1, info
