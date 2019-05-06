#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import _get_schema

from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.api import compss_delete_object

import pandas as pd


def hash_partition(data, settings):
    """
    A Partitioner that generates partitions organized by hash function.

    :param data: A list of pandas DataFrames;
    :param settings: A dictionary with:
     - info:
     - columns: Columns name to perform a hash partition;
     - nfrag: Number of partitions;

    :return: A list of pandas DataFrames;
    """
    info = settings['info'][0]
    nfrag = len(data)
    cols = settings['columns']
    nfrag_target = settings.get('nfrag', nfrag)

    if nfrag_target < 1:
        raise Exception('You must have at least one partition.')

    elif nfrag_target > 1:
        splitted = [[0 for _ in range(nfrag_target)] for _ in range(nfrag)]

        import ddf_library.config
        ddf_library.config.x = nfrag_target
        import ddf_library.functions.etl.repartition as repartition

        for f in range(nfrag):
            splitted[f] = repartition.split_by_hash(data[f], cols,
                                                    info, nfrag_target)

        result = [[] for _ in range(nfrag_target)]
        info = [{} for _ in range(nfrag_target)]
        for f in range(nfrag_target):
            tmp = [splitted[t][f] for t in range(nfrag)]
            result[f] = repartition.merge_n_reduce(concat_10_pandas, tmp, 10)
            info[f] = _get_schema(result[f], f)

        compss_delete_object(splitted)
    else:
        result = data

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@constraint(ComputingUnits="2")  # approach to have more available memory
@task(returns=1)
def concat_10_pandas(df1, df2, df3, df4, df5, df6, df7, df8, df9, df10):
    dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
    dfs = [df for df in dfs if isinstance(df, pd.DataFrame)]
    dfs = pd.concat(dfs, ignore_index=True, sort=False)
    del df1, df2, df3, df4, df5, df6, df7, df8, df9, df10

    return dfs
