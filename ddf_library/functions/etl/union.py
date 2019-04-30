#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from pycompss.api.task import task
import pandas as pd
import numpy as np


def union(data1, data2, settings):
    """
    Function which do a union between two pandas DataFrame.

    :param data1: A list with nfrag pandas's dataframe;
    :param data2: Other list with nfrag pandas's dataframe;
    :param by_name: True to concatenate by column name (Default);
    :param nfrag: New number of partitions (optional, the default
     is the same as data1);
    :return: Returns a list with nfrag pandas's dataframe.

    """

    nfrag1 = len(data1)
    nfrag2 = len(data2)

    by_name = settings.get('by_name', True)
    nfrag = settings.get('nfrag', nfrag1)

    info1 = settings['info'][0]
    info2 = settings['info'][1]

    # first, define the new column names if exists
    cols1 = info1['cols']
    cols2 = info2['cols']
    news_cols1, news_cols2 = [], []

    if not by_name:
        # if we want to concatenate by the columns index, what we need to do
        # is just rename the columns and add new columns to the less one.

        n_cols = max(len(cols2), len(cols1))

        key = 'add'
        news_cols1 = {key: ['col_{}'.format(i) for i in range(n_cols)]}
        news_cols2 = news_cols1
        new_columns = news_cols1[key]

    else:
        diff = set(cols1).difference(set(cols2))
        if len(diff) > 0:
            # in this case, we need to create columns in each dataset

            for col in diff:
                if col not in cols1:
                    news_cols1.append(col)
                else:
                    news_cols2.append(col)

        new_columns = cols1 + news_cols1
        key = 'update'
        news_cols1 = {key: news_cols1}
        news_cols2 = {key: news_cols2}

    # second, update each one
    if len(news_cols1[key]) > 0:
        for f in range(nfrag1):
            data1[f] = _update_cols(data1[f], news_cols1)

    if len(news_cols2[key]) > 0:
        for f in range(nfrag2):
            data2[f] = _update_cols(data2[f], news_cols2)

    # third, group them
    data = data1 + data2

    # and finally, repartition
    new_info = {'cols': new_columns,
                'size': info1['size'] + info2['size']}
    from .repartition import repartition
    repartition_settings = {'info': [new_info], 'nfrag': nfrag}

    output = repartition(data, repartition_settings)

    return output


@task(returns=1)
def _update_cols(data, news_cols):
    if 'update' in news_cols:
        news_cols = news_cols['update']
        for col in news_cols:
            data[col] = np.nan

    elif 'add' in news_cols:
        news_cols = news_cols['add']
        n_cols_init = len(data.columns.tolist())
        n_cols_end = len(news_cols)
        diff = n_cols_end - n_cols_init

        data.columns = news_cols[0:n_cols_init]

        if diff > 0:
            for col in news_cols[n_cols_init:]:
                data[col] = np.nan

    return data

# @task(returns=2)
# def _union(df1, df2, by_name, frag):
#     """Perform a partil union."""
#
#     if len(df1) == 0:
#         result = df2
#     elif len(df2) == 0:
#         result = df1
#     else:
#         if not by_name:
#             o = df2.columns.tolist()
#             n = df1.columns.tolist()
#             diff = len(n) - len(o)
#             if diff < 0:
#                 n = n + o[diff:]
#             elif diff > 0:
#                 n = n[:diff+1]
#             df2.columns = n
#
#         result = pd.concat([df1, df2], ignore_index=True, sort=False)
#
#     info = generate_info(result, frag)
#     return result, info

