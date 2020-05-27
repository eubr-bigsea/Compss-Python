#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT

from ddf_library.utils import generate_info, \
    create_stage_files, read_stage_file, save_stage_file
from ddf_library.functions.etl.repartition import repartition

import pandas as pd
import time


class AddColumnsOperation(object):
    """Merge two DataFrames, column-wise."""

    @staticmethod
    def transform(df1, df2, settings):
        """
        :param df1: A list with nfrag pandas's DataFrame;
        :param df2: A list with nfrag pandas's DataFrame;
        :param settings: a dictionary with:
         - suffixes: Suffixes for attributes (a list with 2 values);
         - schema: a schema (generate automatically by ddf api);
        :return: Returns a list with nfrag pandas's DataFrame.
        """

        suffixes = settings.get('suffixes', ['_l', '_r'])
        info1, info2 = settings['schema'][0], settings['schema'][1]
        len1, len2 = info1['size'], info2['size']
        nfrag1, nfrag2 = len(df1), len(df2)

        swap_cols = sum(len1) < sum(len2)

        if swap_cols:
            params = {'shape': info2['size'], 'schema': [info1]}
            output = repartition(df1, params)
            df1 = output['data']
            nfrag1 = len(df1)
        else:
            params = {'shape': info1['size'], 'schema': [info2]}
            output = repartition(df2, params)
            df2 = output['data']

        # merging two DataFrames with same shape
        result = create_stage_files(nfrag1)
        info = [[] for _ in range(nfrag1)]

        for f in range(nfrag1):
            info[f] = _add_columns(df1[f], df2[f], result[f], suffixes, f)

        output = {'key_data': ['data'], 'key_info': ['schema'],
                  'data': result, 'schema': info}

        return output


@task(returns=1, df1=FILE_IN, df2=FILE_IN, out=FILE_OUT)
def _add_columns(df1, df2, out, suffixes, frag):
    """Perform a partial add columns."""
    t_start = time.time()
    df1 = read_stage_file(df1)
    df2 = read_stage_file(df2)

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    df1 = pd.merge(df1, df2, left_index=True,
                   right_index=True, suffixes=suffixes)
    del df2

    save_stage_file(out, df1)
    info = generate_info(df1, frag)
    t_end = time.time()
    print("[INFO] - Time to process task '{}': {:.0f} seconds"
          .format('_add_columns', t_end - t_start))
    return info
