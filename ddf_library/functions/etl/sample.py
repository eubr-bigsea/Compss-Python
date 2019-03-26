#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
import numpy as np
import math


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

    idxs = _take_define_sample(info, value, nfrag)

    result = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    for f in range(nfrag):
        result[f], info[f] = _get_take(data[f], idxs, f)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


@task(returns=1)
def _take_define_sample(info, head, nfrag):
    """Define the head N indexes to be sampled."""
    n_list = info[2]
    total = sum(n_list)

    if total < head:
        head = total

    list_ids = [[] for _ in range(nfrag)]

    frag = 0
    while head > 0:
        off = head - n_list[frag]
        if off < 0:
            off = head
        else:
            off = n_list[frag]

        list_ids[frag] = off
        head -= off
        frag += 1

    return list_ids


@task(returns=2)
def _get_take(data, indexes, i):
    """Perform a partial sampling."""
    indexes = indexes[i]
    data.reset_index(drop=True, inplace=True)
    result = data.loc[data.index.isin(indexes)]

    info = [result.columns.tolist(), result.dtypes.values, [len(result)]]
    return result, info


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

    TODO: rebalance the list, group the second stage
    """
    nfrag = len(data)
    idxs, seed = _sample_preprocessing(params, nfrag)

    result = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]
    for f in range(nfrag):
        result[f], info[f] = _get_samples(data[f], idxs, f, seed)

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
        idxs = _define_bulks(info, None, seed, True, 'random', nfrag)

    else:
        if op is 'per':
            if value > 1 or value < 0:
                raise Exception('Percentage value must between 0 and 1.0.')

        idxs = _define_bulks(info, value, seed, False, op, nfrag)

    return idxs, seed


@task(returns=1)
def _define_bulks(info, value, seed, random, int_per, nfrag):
    """Define the N random indexes to be sampled."""

    n_list = info[2]
    total = sum(n_list)

    if int_per == 'int':
        if total < value:
            value = total
        ratio = int(math.ceil(float(value)/total))
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
def _get_samples(data, indexes, i, seed):
    """Perform a partial sampling."""

    n = len(data)

    if n > 0:
        value = indexes[i]
        data.reset_index(drop=True, inplace=True)
        data = data.sample(n=value, replace=False, random_state=seed)

    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
    return data, info







