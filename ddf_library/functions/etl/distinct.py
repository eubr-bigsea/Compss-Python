#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, create_stage_files, \
    read_stage_file, save_stage_file

from pycompss.api.parameter import FILE_IN, FILE_OUT
from pycompss.api.task import task

import time


def distinct(data, settings):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.

    :param data: A list with nfrag pandas's DataFrame;
    :param settings: A dictionary with:
     - cols: A list with the columns names to take in count
     (if no field is chosen, all fields are used).
    :return: Returns a list with nfrag pandas's DataFrame.
    """

    out_hash, _ = distinct_stage_1(data, settings)

    nfrag = len(out_hash)
    result = create_stage_files(nfrag)
    info = [[] for _ in range(nfrag)]

    for f in range(nfrag):
        settings['id_frag'] = f
        info[f] = task_distinct_stage_2(out_hash[f], settings.copy(), result[f])

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def distinct_stage_1(data, settings):

    nfrag = len(data)
    cols = settings['columns']
    info1 = settings['info'][0]

    # first, perform a hash partition to shuffle both data
    from .hash_partitioner import hash_partition
    if settings.get('opt_function', False):
        info1['function'] = [distinct_stage_2, {'columns': cols, 'id_frag': -1}]
    hash_params = {'columns': cols, 'nfrag': nfrag, 'info': [info1]}
    output1 = hash_partition(data, hash_params)
    out_hash = output1['data']

    return out_hash, settings


def distinct_stage_2(data, settings):
    """Remove duplicate rows."""

    cols = settings['columns'] if settings['columns'] else []
    frag = settings['id_frag']

    if len(cols) == 0:
        cols = list(data.columns)

    data = data.drop_duplicates(cols, keep='first', ignore_index=True)
    info = generate_info(data, frag)

    return data, info


@task(returns=1, data_input=FILE_IN, data_output=FILE_OUT)
def task_distinct_stage_2(data_input, settings, data_output):
    t_start = time.time()
    data = read_stage_file(data_input)
    result, info = distinct_stage_2(data, settings)
    save_stage_file(data_output, result)

    t_end = time.time()
    print("[INFO] - Time to process task '{}': {:.0f} seconds"
          .format('_balancer_get_rows', t_end - t_start))
    return info
