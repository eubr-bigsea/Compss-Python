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
     the distribution; and the second is to create the sample itself. The first
     part cannot be grouped with others tasks because this function needs
     the schema information. The second part could be grouped.

    TODO: rebalance the list, group the second stage
    """
    nfrag = len(data)

    info = settings['info'][0]
    value = settings['value']

    idxs = _take_define_sample(info, value)

    result = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    for f in range(nfrag):
        result[f], info[f] = _get_take(data[f], idxs, f)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


# @task(returns=1)
def _take_define_sample(info, head):
    """Define the head N indexes to be sampled."""
    info = compss_wait_on(info)
    n_list = info['size']
    total = sum(n_list)

    if total < head:
        head = total

    cumsum = np.cumsum(n_list)
    idx = next(x for x, val in enumerate(cumsum) if val >= head)

    list_ids = n_list[0: idx+1]
    list_ids[-1] -= (cumsum[idx] - head)

    diff = len(n_list) - len(list_ids)
    diff = [0 for _ in range(diff)]
    list_ids += diff

    return list_ids


@task(returns=2)
def _get_take(data, indexes, frag):
    """Perform a partial sampling."""
    indexes = indexes[frag]
    data.reset_index(drop=True, inplace=True)
    result = data.iloc[:indexes]
    del data
    info = generate_info(result, frag)
    return result, info
