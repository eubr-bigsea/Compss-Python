#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.api.parameter import FILE_INOUT

from ddf_library.utils import read_stage_file, save_stage_file
import numpy as np


def union(data1, data2, settings):
    """
    Function which do a union between two pandas DataFrame. Union can only be
    performed on tables with the same number of columns.

    :param data1: A list with nfrag pandas's DataFrame;
    :param data2: Other list with nfrag pandas's DataFrame;
    :param settings: A dictionary with:
     - by_name: True to concatenate by column name (Default);
     - nfrag: New partition number (optional, the default is the same as data1);
    :return: Returns a list with nfrag pandas's DataFrame.
    """

    nfrag1, nfrag2 = len(data1), len(data2)

    by_name = settings.get('by_name', True)
    nfrag = settings.get('nfrag', nfrag1)

    info1, info2 = settings['schema']

    # first, define the new column names if exists
    cols1, cols2 = info1['cols'], info2['cols']
    news_cols1, news_cols2 = [], []

    if not by_name:
        if len(cols2) != len(cols1):
            raise Exception("Union can only be performed on tables with the "
                            "same number of columns.")

        key = 'add'
        news_cols1 = {key: cols1}
        news_cols2 = news_cols1
        new_columns = news_cols1[key]

    else:
        diff = set(cols1).difference(set(cols2))
        if len(diff) > 0:
            # in this case, we need to create columns in each data set

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
            _update_cols(data1[f], news_cols1)

    if len(news_cols2[key]) > 0:
        for f in range(nfrag2):
            _update_cols(data2[f], news_cols2)

    # third, group them
    data = data1 + data2

    # and finally, repartition
    new_info = {'cols': new_columns,
                'size': info1['size'] + info2['size']}
    from .repartition import repartition
    repartition_settings = {'schema': [new_info], 'nfrag': nfrag}

    output = repartition(data, repartition_settings)

    return output


@task(data_input=FILE_INOUT)
def _update_cols(data_input, news_cols):

    data = read_stage_file(data_input)
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

    save_stage_file(data_input, data)
