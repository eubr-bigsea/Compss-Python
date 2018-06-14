#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np
import math


class SampleOperation(object):
    """Sample Operation.

    Returns a sampled subset of the input panda's dataFrame.
    """

    def transform(self, data, params, nfrag):
        """SampleOperation.

        :param data: A list with nfrag pandas's dataframe;
        :param params: dictionary that contains:
            - type:
                * 'percent': Sample a random amount of records (default)
                * 'value': Sample a N random records
                * 'head': Sample the N firsts records of the dataframe
            - seed : Optional, seed for the random operation.
            - int_value: Value N to be sampled (in 'value' or 'head' type)
            - per_value: Percentage to be sampled (in 'value' or 'head' type)
        :param nfrag: The number of fragments;
        :return: A list with nfrag pandas's dataframe.
        """

        idxs = self.preprocessing(params, data, nfrag)

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _get_samples(data[f], idxs, f)

        return result

    def preprocessing(self, params, data, nfrag):
        """Check the settings."""
        TYPE = params.get("type", 'percent')
        if TYPE not in ['percent', 'value', 'head']:
            raise Exception('You must inform a valid sampling type.')

        value = -1
        op = 'int'
        if TYPE == 'head' or TYPE == 'value':
            if 'int_value' in params:
                value = params['int_value']
                op = 'int'
                if not isinstance(value, int) and value < 0:
                    raise Exception('`int_value` must be a positive integer.')
            elif 'per_value' in params:
                value = params['per_value']
                op = 'per'
                if value > 1 or value < 0:
                    raise Exception('Percentage value must between 0 and 1.0.')
            else:
                raise Exception('Using `Head` or `value` sampling type you '
                                'need to set `int_value` or `per_value` '
                                'setting as well.')

        partial_counts = [_count_records(data[i]) for i in range(nfrag)]
        info = mergeReduce(_merge_counts, partial_counts)

        if TYPE == 'percent':
            seed = params.get('seed', None)
            idxs = _define_n_sample(info, None, seed, True, 'null', nfrag)

        elif TYPE == 'value':
            seed = params.get('seed', None)
            idxs = _define_n_sample(info, value, seed, False, op, nfrag)

        elif TYPE == 'head':
            idxs = _define_head_sample(info, value, op, nfrag)

        return idxs

    def transform_serial(self, data, indexes, i):
        """Perform a partial sampling."""
        indexes = indexes[i]
        data = data.reset_index(drop=True)
        sample = data.loc[data.index.isin(indexes)]

        return [sample, len(sample)]


@task(returns=list)
def _count_records(data):
    """It is necessary count the distribuition of the data in each fragment."""
    size = len(data)
    return [size, [size]]


@task(returns=list)
def _merge_counts(data1, data2):
    """Merge the partial counts."""
    return [data1[0]+data2[0], np.concatenate((data1[1], data2[1]), axis=0)]


@task(returns=list)
def _define_n_sample(n_list, value, seed, random, int_per, nfrag):
    """Define the N random indexes to be sampled."""

    total, n_list = n_list
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


@task(returns=list)
def _define_head_sample(n_list, head, int_per, nfrag):
    """Define the head N indexes to be sampled."""
    total, n_list = n_list
    if int_per == 'int':
        if total < head:
            head = total
    elif int_per == 'per':
        head = int(math.ceil(total*head))

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


@task(returns=list)
def _get_samples(data, indexes, i):
    """Perform a partial sampling."""
    indexes = indexes[i]
    data.reset_index(drop=True, inplace=True)
    return data.loc[data.index.isin(indexes)]



