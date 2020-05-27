#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.functions.reduce import merge_reduce

from ddf_library.utils import generate_info, read_stage_file, create_stage_files

import numpy as np
import pandas as pd


def cross_tab(data, settings):
    """
    Computes a pair-wise frequency table of the given columns. Also known as
    a contingency table. The number of distinct values for each column should
    be less than 1e4. At most 1e6 non-zero pair frequencies will be returned.

    :param data: A list of pandas's DataFrame;
    :param settings: A dictionary that contains:
        - 'col1': The name of the first column
        - 'col2': The name of the second column
    :return: A list of pandas's DataFrame;
    """

    cols = [settings['col1'], settings['col2']]
    nfrag = len(data)

    partial = [_crosstab_partial(data[f], cols) for f in range(nfrag)]
    crosstab_df = merge_reduce(_merge_counts, partial)
    compss_delete_object(partial)

    data_output = create_stage_files(nfrag)
    info = _create_tab(crosstab_df, nfrag, data_output)

    output = {'key_data': ['data'], 'key_info': ['schema'],
              'data': data_output, 'schema': info}
    return output


@task(returns=1, data_input=FILE_IN)
def _crosstab_partial(data_input, cols):
    col1, col2 = cols
    data = read_stage_file(data_input, cols)

    data = pd.crosstab(index=data[col1], columns=data[col2])
    data.columns = data.columns.values
    data.index = data.index.values
    return data


@task(returns=1)
def _merge_counts(data1, data2):

    max_size_cols = 1e4
    max_len_rows = 1e6

    data = data1.add(data2, fill_value=0).fillna(0).astype(int)
    size_cols = data.shape[1]
    size_len = data.shape[0]

    if size_cols > max_size_cols:
        data = data.drop(data.columns[max_size_cols:], axis=1)

    if size_len > max_len_rows:
        data = data[:max_len_rows]

    return data


def _create_tab(data, nfrag, data_ouput):
    data = compss_wait_on(data)
    data.insert(0, 'key', data.index.values)

    data = np.array_split(data, nfrag)
    info = [generate_info(data[f], f) for f in range(nfrag)]
    for df, out in zip(data, data_ouput):
        df.to_parquet(out)

    return info
