#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info
from pycompss.api.task import task
import numpy as np
import math


def sample(data, params):
    """
    Returns a sampled subset of the input panda's DataFrame.

    :param data: A list of pandas's DataFrame;
    :param params: dictionary that contains:
      - type: 'percent' to sample a random amount of records (default) or
       'value' to sample a N random records;
      - seed : Optional, seed for the random operation.
      - int_value: Value N to be sampled (in 'value' or 'head' type)
      - per_value: Percentage to be sampled (in 'value' or 'head' type)
      - info: information generated from others tasks (automatic);
    :return: A list of pandas's DataFrame.

    .. note: This operations contains two stages: the first one is to define
     the distribution; and the second is to create the sample itself. The first
     part cannot be grouped with others tasks because this function needs
     the schema information. The second part could be grouped.

    TODO: re-balance the list, group the second stage
    """
    nfrag = len(data)
    idx_list, seed = _sample_preprocessing(params, nfrag)

    result = [[] for _ in range(nfrag)]
    info = result[:]
    for f in range(nfrag):
        result[f], info[f] = _get_samples(data[f], idx_list, f, seed)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


def _sample_preprocessing(params, nfrag):
    """Check the settings."""
    sample_type = params.get("type", 'percent')
    seed = params.get('seed', None)
    info = params['info'][0]
    value = abs(params['value'])
    op = 'int' if isinstance(value, int) else 'per'

    if sample_type == 'percent':
        idx_list = _define_bulks(info, None, seed, True, 'random', nfrag)

    else:
        if op is 'per':
            if value > 1 or value < 0:
                raise Exception('Percentage value must between 0 and 1.0.')

        idx_list = _define_bulks(info, value, seed, False, op, nfrag)

    return idx_list, seed


@task(returns=1)
def _define_bulks(info, value, seed, random, int_per, nfrag):
    """Define the N random indexes to be sampled."""

    n_list = info['size']
    total = sum(n_list)

    if int_per == 'int':
        if total < value:
            value = total

        if total == 0:
            ratio = 0
        else:
            ratio = value/total
    else:
        ratio = value

    if random:
        np.random.seed(seed)
        ratio = np.random.random_sample()

    target_value = int(math.ceil(ratio * total))
    np.random.seed(seed)
    sizes = [int(math.ceil(n * ratio)) for n in n_list]

    val = sum(sizes)
    for i in range(nfrag):
        if val == target_value:
            break
        if sizes[i] > 0:
            sizes[i] -= 1
            val -= 1

    return sizes


@task(returns=2)
def _get_samples(data, indexes, frag, seed):
    """Perform a partial sampling."""

    n = len(data)

    if n > 0:
        data.reset_index(drop=True, inplace=True)
        data = data.sample(n=indexes[frag], replace=False, random_state=seed)

    info = generate_info(data, frag)
    return data, info
