#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, read_stage_file, \
    create_stage_files, save_stage_file, merge_info, delete_result
from ddf_library.functions.etl.balancer import WorkloadBalancer
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import FILE_IN, FILE_OUT

import numpy as np


def take(data, settings):
    """
    Returns the first n elements of the input panda's DataFrame.

    :param data: A list of pandas's DataFrame;
    :param settings: dictionary that contains:
     - value: integer value to be sampled;
     - info: information generated from others tasks (automatic);
    :return: A list of pandas's DataFrame.

    .. note: This operations contains two stages: the first one is to define
     the distribution; and the second is to create the sample itself.
    """

    balancer = settings.get('balancer', True)
    data, settings = take_stage_1(data, settings)

    nfrag = len(data)
    info = [[] for _ in range(nfrag)]
    result = create_stage_files(nfrag)

    for f in range(nfrag):
        settings['id_frag'] = f
        info[f] = task_take_stage_2(data[f], result[f], settings.copy())

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}

    if balancer:
        info = merge_info(info)
        conf = {'forced': True, 'info': [info]}
        output = WorkloadBalancer(conf).transform(result)
        delete_result(result)

    return output


def take_stage_1(data, settings):

    info, value = settings['info'][0], settings['value']
    settings['idx'] = _take_define_sample(info, value)
    settings['intermediate_result'] = False
    return data, settings


def _take_define_sample(info, head):
    """Define the head N indexes to be sampled."""
    info = compss_wait_on(info)
    n_list = info['size']
    total = sum(n_list)

    if total < head:
        head = total

    cum_sum = np.cumsum(n_list)
    idx = next(x for x, val in enumerate(cum_sum) if val >= head)

    list_ids = n_list[0: idx+1]
    list_ids[-1] -= (cum_sum[idx] - head)

    diff = len(n_list) - len(list_ids)
    diff = [0 for _ in range(diff)]
    list_ids += diff

    return list_ids


def take_stage_2(data, settings):
    """Perform a partial sampling."""
    frag = settings['id_frag']
    indexes = settings['idx'][frag]
    del settings

    data.reset_index(drop=True, inplace=True)
    data = data.iloc[:indexes]
    info = generate_info(data, frag)
    return data, info


@task(returns=1, input_data=FILE_IN, output_data=FILE_OUT)
def task_take_stage_2(input_data, output_data, settings):
    data = read_stage_file(input_data)
    data, info = take_stage_2(data, settings)
    save_stage_file(output_data, data)
