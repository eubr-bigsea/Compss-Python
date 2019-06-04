#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.utils import generate_info
from pycompss.api.task import task
import math


def random_split(data, settings):
    """
    Randomly splits a DataFrame into two distinct DataFrames.

    :param data: A list with nfrag pandas's DataFrame;
    :param settings: A dictionary that contains:
      - 'percentage': Percentage to split the data (default, 0.5);
      - 'seed': Optional, seed in case of deterministic random operation;
      - 'info': information generated from others tasks (automatic);
    :return: Returns two lists with nfrag pandas's DataFrame with distinct
     subsets of the input.

    ..note: The operation consists of two stages: first, we define the
     distribution; and the second we split the data. The first part cannot be
     grouped with others stages because we need information about the
     size in each partition. The second part cannot be grouped as well because
     generates two outputs data.
    """
    nfrag = len(data)

    idx = _preprocessing(settings, nfrag)
    out1 = [[] for _ in range(nfrag)]
    out2, info1, info2 = out1[:], out1[:], out1[:]

    for i, fraction in enumerate(idx):
        out1[i], info1[i], out2[i], info2[i] = _split_get(data[i], fraction, i)

    output = {'key_data': ['data1', 'data2'],
              'key_info': ['info1', 'info2'],
              'data1': out1, 'info1': info1,
              'data2': out2, 'info2': info2}
    return output


def _preprocessing(settings, nfrag):
    percentage = settings.get('percentage', 0.5)
    info = settings['info'][0]
    n_list = info['size']

    if percentage < 0 or percentage > 1:
        raise Exception("Please inform a valid percentage [0, 1].")

    idx_list = _split_allocate(n_list, percentage, nfrag)

    return idx_list


def _split_allocate(n_list, fraction, nfrag):
    """Define a list of indexes to be divided."""

    n_rows = sum(n_list)

    size = int(math.ceil(n_rows*fraction))
    sizes = [int(math.ceil(n * fraction)) for n in n_list]

    val = sum(sizes)
    while val != size:
        for i in range(nfrag):
            if val == size:
                break
            if sizes[i] > 0:
                sizes[i] -= 1
                val -= 1

    return sizes


@task(returns=4)
def _split_get(data, value, frag):
    """Retrieve the split."""

    n = len(data)

    if n > 0:
        data = data.sample(frac=1).reset_index(drop=True)

        split2 = data.iloc[value:].reset_index(drop=True)
        data = data.iloc[:value].reset_index(drop=True)

    else:
        split2 = data.copy()

    info1 = generate_info(data, frag)
    info2 = generate_info(split2, frag)

    return data, info1, split2, info2
