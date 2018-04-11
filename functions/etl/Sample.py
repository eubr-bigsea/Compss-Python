#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
import numpy as np
import math


class SampleOperation(object):
    """Sample Operation.

    Returns a sampled subset of the input panda's dataFrame.
    """

    def transform(self, data, params, nFrag):
        """SampleOperation.

        :param data: A list with numFrag pandas's dataframe;
        :param params: dictionary that contains:
            - type:
                * 'percent': Sample a random amount of records (default)
                * 'value': Sample a N random records
                * 'head': Sample the N firsts records of the dataframe
            - seed : Optional, seed for the random operation.
            - int_value: Value N to be sampled (in 'value' or 'head' type)
            - per_value: Percentage to be sampled (in 'value' or 'head' type)
        :param nFrag: The number of fragments;
        :return: A list with numFrag pandas's dataframe.
        """
        value, int_per = self.validate(params)
        TYPE = params.get("type", 'percent')
        if TYPE not in ['percent', 'value', 'head']:
            raise Exception('Please inform a valid type mode')

        partial_counts = [self.count_records(data[i]) for i in range(nFrag)]
        N_list = mergeReduce(self.mergeCount, partial_counts)

        result = [[] for i in range(nFrag)]
        if TYPE == 'percent':
            seed = params.get('seed', None)
            idxs = self.define_n_sample(N_list, None, seed,
                                        True, 'null', nFrag)

        elif TYPE == 'value':
            seed = params.get('seed', None)
            idxs = self.define_n_sample(N_list, value, seed,
                                        False, int_per, nFrag)

        elif TYPE == 'head':
            idxs = self.define_head_sample(N_list, value, int_per, nFrag)

        for i in range(nFrag):
            result[i] = self.get_samples(data[i], idxs, i)

        return result

    def validate(self, params):
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

        return value, op

    @task(isModifier=False, returns=list)
    def count_records(self, data):
        """Count the distribuition of the data in each fragment."""
        size = len(data)
        return [size, [size]]

    @task(isModifier=False, returns=list)
    def mergeCount(self, df1, df2):
        """Merge the partial counts."""
        return [df1[0]+df2[0], np.concatenate((df1[1], df2[1]), axis=0)]

    @task(isModifier=False, returns=list)
    def define_n_sample(self, N_list, value, seed, random, int_per, numFrag):
        """Define the N random indexes to be sampled."""
        total, n_list = N_list
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
        list_ids = [[] for i in range(numFrag)]

        first_id = 0
        for i in range(numFrag):
            last_id = sizes[i]
            idx = (ids >= first_id) & (ids < last_id)
            list_ids[i] = ids[idx] - first_id
            first_id = last_id

        return list_ids

    @task(isModifier=False, returns=list)
    def define_head_sample(self, N_list, head, int_per, numFrag):
        """Define the head N indexes to be sampled."""
        total, n_list = N_list

        if int_per == 'int':
            if total < head:
                head = total
        elif int_per == 'per':
            head = int(math.ceil(total*head))

        list_ids = [[] for i in range(numFrag)]

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

    @task(isModifier=False, returns=list)
    def get_samples(self, data, indexes, i):
        """Perform a partial sampling."""
        indexes = indexes[i]
        data = data.reset_index(drop=True)
        sample = data.loc[data.index.isin(indexes)]

        return sample
