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
        result[f], info[f] = _get_samples(data[f], idxs, f)

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


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
    idxs = _sample_preprocessing(params, nfrag)

    result = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]
    for f in range(nfrag):
        result[f], info[f] = _get_samples(data[f], idxs, f)

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
        idxs = _define_n_sample(info, None, seed, True, 'null', nfrag)

    else:
        if op is 'per':
            if value > 1 or value < 0:
                raise Exception('Percentage value must between 0 and 1.0.')

        idxs = _define_n_sample(info, value, seed, False, op, nfrag)

    return idxs


@task(returns=1)
def _define_n_sample(info, value, seed, random, int_per, nfrag):
    """Define the N random indexes to be sampled."""

    n_list = info[2]
    total = sum(n_list)

    if int_per == 'int':
        if total < value:
            value = total
    elif int_per == 'per':
        value = int(math.ceil(total*value))

    if random:
        np.random.seed(seed)
        percentage = np.random.random_sample()
        value = int(math.ceil(total*percentage))

    np.random.seed(seed)
    ids = np.array(sorted(np.random.choice(total, value, replace=False)))
    sizes = np.cumsum(n_list)
    list_ids = [[] for _ in range(nfrag)]

    first_id = 0
    for i in range(nfrag):
        last_id = sizes[i]
        idx = (ids >= first_id) & (ids < last_id)
        list_ids[i] = ids[idx] - first_id
        first_id = last_id

    return list_ids


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

        list_ids[frag] = [i for i in range(off)]
        head -= off
        frag += 1

    return list_ids


@task(returns=2)
def _get_samples(data, indexes, i):
    """Perform a partial sampling."""
    indexes = indexes[i]
    data.reset_index(drop=True, inplace=True)
    result = data.loc[data.index.isin(indexes)]

    info = [result.columns.tolist(), result.dtypes.values, [len(result)]]
    return result, info



