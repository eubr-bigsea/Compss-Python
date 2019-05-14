#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from pycompss.api.task import task
import pandas as pd


def distinct_stage_1(data, settings):
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

    return out_hash, settings


def distinct_stage_2(data, settings):
    """Remove duplicate rows."""

    cols = settings['columns']
    frag = settings['id_frag']

    all_cols = data.columns
    if len(cols) == 0:
        cols = all_cols

    data = data.drop_duplicates(cols, keep='first').reset_index(drop=True)
    info = generate_info(data, frag)

    return data, info


@task(returns=2)
def task_distinct_stage_2(data, settings):
    return distinct_stage_2(data, settings)
