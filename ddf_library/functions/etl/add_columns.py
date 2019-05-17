#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task

from ddf_library.utils import generate_info
from ddf_library.functions.etl.repartition import repartition

import pandas as pd


class AddColumnsOperation(object):
    """Merge two DataFrames, column-wise."""

    @staticmethod
    def transform(df1, df2, settings):
        """
        :param df1: A list with nfrag pandas's DataFrame;
        :param df2: A list with nfrag pandas's DataFrame;
        :param settings: a dictionary with:
         - suffixes: Suffixes for attributes (a list with 2 values);
         - info: a schema (generate automatically by ddf api);
        :return: Returns a list with nfrag pandas's DataFrame.
        """

        suffixes = settings.get('suffixes', ['_l', '_r'])
        info1, info2 = settings['info'][0], settings['info'][1]
        len1, len2 = info1['size'], info2['size']
        nfrag1, nfrag2 = len(df1), len(df2)

        swap_cols = sum(len1) < sum(len2)

        if swap_cols:
            params = {'shape': info2['size'], 'info': [info1]}
            output = repartition(df1, params)
            df1 = output['data']
            nfrag1 = len(df1)
        else:
            params = {'shape': info1['size'], 'info': [info2]}
            output = repartition(df2, params)
            df2 = output['data']

        # merging two DataFrames with same shape
        result = [[] for _ in range(nfrag1)]
        info = [[] for _ in range(nfrag1)]

        for f in range(nfrag1):
            result[f], info[f] = _add_columns(df1[f], df2[f], suffixes, f)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}

        # output = {'df1': df1, 'df2': df2, 'suffixes': suffixes}

        return output


@task(returns=2)
def _add_columns(df1, df2, suffixes, frag):
    """Perform a partial add columns."""

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    df1 = pd.merge(df1, df2, left_index=True,
                   right_index=True, suffixes=suffixes)
    del df2

    df1.reset_index(drop=True, inplace=True)
    info = generate_info(df1, frag)
    return df1, info
