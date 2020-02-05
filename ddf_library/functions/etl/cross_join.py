#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, create_auxiliary_column, \
    create_stage_files, read_stage_file, save_stage_file
from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_INOUT

import pandas as pd
import time


def cross_join(data1, data2):
    """
    Returns the cartesian product with another DataFrame.

    :param data1: A list of pandas's DataFrame;
    :param data2: A list of pandas's DataFrame;
    :return: Returns a list of pandas's DataFrame.
    """

    nfrag = len(data1)
    result = [[] for _ in range(nfrag)]
    info = result[:]

    result = create_stage_files(nfrag)

    for f, df1 in enumerate(data1):
        pd.DataFrame([]).to_parquet(result[f])
        for df2 in data2:
            info[f] = _cross_join(result[f], df1, df2, f)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@task(returns=1, result=FILE_INOUT, df1=FILE_IN, df2=FILE_IN)
def _cross_join(result, df1, df2, frag):
    t1 = time.time()
    df = read_stage_file(result)
    df1 = read_stage_file(df1)
    df2 = read_stage_file(df2)
    key = create_auxiliary_column(df1.columns.tolist() + df2.columns.tolist())

    df1[key] = 1
    df2[key] = 1

    product = df1.merge(df2, on=key).drop(key, axis=1)

    if len(df) == 0:
        df = product
    else:
        df = pd.concat([df, product], sort=False)

    info = generate_info(df, frag)
    save_stage_file(result, df)
    t2 = time.time()
    print('[INFO] - Time to process the complete stage: {:.0f}'.format(t2 - t1))
    return info
