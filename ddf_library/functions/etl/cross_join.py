#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from pycompss.api.task import task
import pandas as pd
import string
import random


def crossjoin(data1, data2):
    """
    Returns the cartesian product with another DataFrame.

    :param data1: A list of pandas's dataframe;
    :param data2: A list of pandas's dataframe;
    :return: Returns a list of pandas's dataframe.
    """

    nfrag = len(data1)
    result = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    for f, df1 in enumerate(data1):
        for df2 in data2:
            result[f], info[f] = _crossjoin(result[f], df1, df2, f)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@task(returns=2)
def _crossjoin(result, df1, df2, frag):

    key = 'key'
    while key in df1.columns or key in df2.columns:
        key = ''.join([random.choice(string.ascii_letters) for _ in xrange(10)])

    df1[key] = 1
    df2[key] = 1

    product = df1.merge(df2, on=key).drop(key, axis=1)

    if len(result) == 0:
        result = product
    else:
        result = pd.concat([result, product], sort=True)

    info = generate_info(result, frag)
    return result, info
