#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on

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

    TODO: re-balance the list, group the second stage
    TODO: Use schema to avoid submit empty tasks
    """

    data, settings = take_stage_1(data, settings)

    nfrag = len(data)
    result = [[] for _ in range(nfrag)]
    info = result[:]

    for f in range(nfrag):
        settings['id_frag'] = f
        result[f], info[f] = task_take_stage_2(data[f], settings)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def take_stage_1(data, settings):

    info, value = settings['info'][0], settings['value']

    settings['idx'] = _take_define_sample(info, value)

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
    result = data.iloc[:indexes]

    del data
    info = generate_info(result, frag)
    return result, info


@task(returns=2)
def task_take_stage_2(data, settings):
    return take_stage_2(data, settings)