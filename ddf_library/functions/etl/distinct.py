#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from pycompss.api.task import task
import pandas as pd


def distinct(data, settings):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.

    :param data: A list with nfrag pandas's DataFrame;
    :param settings: A dictionary with:
     - cols: A list with the columns names to take in count
     (if no field is chosen, all fields are used).
    :return: Returns a list with nfrag pandas's DataFrame.
    """
    nfrag = len(data)
    cols = settings['columns']
    info1 = settings['info'][0]

    # first, perform a hash partition to shuffle both data
    from .hash_partitioner import hash_partition
    hash_params = {'columns': cols, 'nfrag': nfrag, 'info': [info1]}
    output1 = hash_partition(data, hash_params)
    out_hash = output1['data']

    result = [[] for _ in range(nfrag)]
    info = result[:]

    for f in range(nfrag):
        result[f], info[f] = _drop_duplicates(out_hash[f], cols, f)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@task(returns=2)
def _drop_duplicates(data, cols, frag):
    """Remove duplicate rows."""

    all_cols = data.columns
    if len(cols) == 0:
        cols = all_cols

    data = data.drop_duplicates(cols, keep='first').reset_index(drop=True)
    info = generate_info(data, frag)

    return data, info
