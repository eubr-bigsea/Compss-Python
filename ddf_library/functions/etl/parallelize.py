#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.functions.data import chunks
from pycompss.api.api import compss_wait_on

import math
import pandas as pd


def parallelize(data, nfrag):
    """
       Method to split the data in nfrag parts. This method simplifies
       the use of chunks.

       :param data: The np.array or list to do the split.
       :param nfrag: A number of partitions
       :return The array splitted.

       :Note: the result may be unbalanced when the number of rows is too small
    """

    new_size = int(math.ceil(float(len(data))/nfrag))
    result = [d for d in chunks(data, new_size)]

    while len(result) < nfrag:
        result.append(pd.DataFrame(columns=result[0].columns))

    info = []
    for r in result:
        info.append([r.columns.tolist(), r.dtypes.values, [len(r)]])
    if len(result) > nfrag:
        raise Exception("Error in parallelize function")

    return result, info


def import_to_ddf(df_list):
    """
    In order to import a list of DataFrames in DDF abstraction, we need to
    check the schema of each partition.

    :param df_list: a List of Pandas DataFrames
    :return: a List of Pandas DataFrames and a schema
    """

    nfrag = len(df_list)
    result = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    for f in range(nfrag):
        result[f], info[f] = _get_schema(df_list[f])

    info = merge_reduce(merge_schema, info)
    info = compss_wait_on(info)

    columns, dtypes, n = 0, 0, 0
    for f, schema in enumerate(info):
        if f == 0:
            columns, dtypes, n = schema
        else:
            columnsf, dtypefs, nf = schema
            if set(columns) != set(columnsf):
                raise Exception("Partitions have different columns names.")
            n += nf
            # Check different datatypes

    info = [columns, dtypes, n]
    return result, info


@task(returns=2)
def _get_schema(df):

    info = [df.columns.tolist(), df.dtypes.values, [len(df)]]
    return df, [info]


@task(returns=1, priority=True)
def merge_schema(schema1, schema2):
    return schema1 + schema2
