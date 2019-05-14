#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, create_auxiliary_column
from pycompss.api.task import task
import pandas as pd


def except_all(data1, data2, settings):
    """
    Return a new DataFrame containing rows in this DataFrame but not in
    another DataFrame while preserving duplicates. This is equivalent to
    EXCEPT ALL in SQL.

    :param data1: A list of pandas's DataFrame;
    :param data2: The second list of pandas's DataFrame;
    :param settings: A empty dictionary with:
     - 'info':
    :return: A list of pandas's DataFrame.
    """
    data1, data2, _ = except_all_stage_1(data1, data2, settings)

    nfrag = len(data1)
    info = [[] for _ in range(nfrag)]
    result = info[:]

    for f in range(nfrag):
        settings['id_frag'] = f
        result[f], info[f] = except_all_stage_2(data1[f], data2[f], settings)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def except_all_stage_1(data1, data2, settings):
    info1, info2 = settings['info']
    nfrag = len(data1)

    from .hash_partitioner import hash_partition
    params_hash1 = {'columns': [], 'info': [info1], 'nfrag': nfrag}
    out1 = hash_partition(data1, params_hash1)
    data1 = out1['data']

    params_hash2 = {'columns': [], 'info': [info2], 'nfrag': nfrag}
    out2 = hash_partition(data2, params_hash2)
    data2 = out2['data']

    return data1, data2, settings


def except_all_stage_2(df1, df2, settings):
    """Perform a Difference partial operation keeping duplicated rows."""
    frag = settings['id_frag']

    name1, name2 = list(df1.columns), list(df2.columns)
    if len(df1) > 0 and len(df2) > 0:

        check_cols = all([True for col in name1 if col in name2])
        if len(name1) == len(name2) and check_cols:

            # count frequency of each row in both data set
            df1 = df1.groupby(name1).size()
            df2 = df2.groupby(name1).size()

            # subtract the frequency and keep only rows with frequency > 0
            df1 = df1.subtract(df2, fill_value=0).loc[lambda x: x > 0]

            # convert the series to DataFrame
            df1 = df1.to_frame().reset_index()

            col_aux = create_auxiliary_column(name1)
            df1.columns = name1 + [col_aux]

            df1 = df1\
                .reindex(df1.index.repeat(df1[col_aux]))\
                .drop([col_aux], axis=1)

    info = generate_info(df1, frag)
    return df1, info


@task(returns=2)
def task_except_all_stage_2(data1, data2, settings):
    return except_all_stage_2(data1, data2, settings)
