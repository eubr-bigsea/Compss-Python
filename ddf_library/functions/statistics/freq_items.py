#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
# from pycompss.api.local import local # guppy module isnt available in python3
from pycompss.api.api import compss_delete_object, compss_wait_on

from ddf_library.utils import read_stage_file

import numpy as np
import pandas as pd


def freq_items(data, settings):
    """
     Finding frequent items for columns, possibly with false positives. Using
     the frequent element count algorithm described in
    "https://doi.org/10.1145/762471.762473, proposed by Karp, Schenker, and
    Papadimitriou".

    :param data: A list of pandas's DataFrame;
    :param settings: A dictionary that contains:
        - 'col': Names of the columns to search frequent items in
        - 'support': The minimum frequency for an item to be considered
        `frequent`. Should be greater than 1e-4.
    :return: A DataFrame
    """

    nfrag = len(data)
    support = settings['support']
    if support is None:
        support = 0.01
    settings['support'] = int(1/support)

    if not isinstance(settings['col'], list):
        settings['col'] = [settings['col']]

    partial_frequency = [[] for _ in range(nfrag)]
    for f in range(nfrag):
        partial_frequency[f] = _freq_items(data[f], settings)

    global_frequency = merge_reduce(_freq_items_merge, partial_frequency)
    compss_delete_object(partial_frequency)
    frequent_items = _freq_items_get(global_frequency)

    return frequent_items


@task(returns=1, data_input=FILE_IN)
def _freq_items(data_input, settings):
    col = settings['col']
    support = settings['support']
    freq = {}

    df = read_stage_file(data_input, col)

    for c in col:
        base_map = {}
        values = df[c].values
        for key in values:
            if key != np.nan:
                if key in base_map:
                    base_map[key] += 1
                else:
                    base_map[key] = 1

                    if len(base_map) > support:
                        for k in list(base_map.keys()):
                            base_map[k] -= 1
                            if base_map[k] == 0:
                                del base_map[k]
        freq[c] = base_map

    return [freq, settings]


@task(returns=1)
def _freq_items_merge(df1, df2):
    freq1, settings = df1
    freq2, _ = df2

    support = settings['support']

    for c in settings['col']:
        base_map = freq1[c]

        for k2 in freq2[c]:
            count = freq2[c][k2]
            if k2 in base_map:
                base_map[k2] += count
            else:
                if len(base_map) < support:
                    base_map[k2] = count
                else:
                    if len(base_map) > 0:
                        _, min_count = min(base_map.items(), key=lambda x: x[1])
                    else:
                        min_count = 0

                    if (count - min_count) >= 0:
                        base_map[k2] = count
                        base_map = {k: v - min_count
                                    for k, v in base_map.items()
                                    if v > min_count}
                    else:
                        base_map = {k: v - count for k, v in base_map.items()}
        freq1[c] = base_map

    return [freq1, settings]


# @local
def _freq_items_get(info):
    info = compss_wait_on(info)
    freq, settings = info
    cols = settings['col']

    cols_name = ['{}_freqItems'.format(c) for c in cols]
    row = [freq[c].keys() for c in cols]
    freq = pd.DataFrame([row], columns=cols_name)

    return freq
