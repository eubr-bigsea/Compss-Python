#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.utils import generate_info, create_stage_files, \
    save_stage_file, read_stage_file
from pycompss.api.task import task
from pycompss.api.api import compss_open
from pycompss.api.parameter import FILE_IN, FILE_OUT
import math


def random_split(input_files, settings):
    """
    Randomly splits a DataFrame into two distinct DataFrames.

    :param input_files: A list with nfrag pandas's DataFrame;
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
    nfrag = len(input_files)

    idx = _preprocessing(settings, nfrag)
    info1 = [[] for _ in range(nfrag)]
    info2 = info1[:]

    out1 = create_stage_files(nfrag)
    out2 = create_stage_files(nfrag)
    for i, fraction in enumerate(idx):
        info1[i], info2[i] = _split_get(input_files[i], out1[i],
                                        out2[i], fraction, i)

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


@task(returns=2, data=FILE_IN, fout1=FILE_OUT, fout2=FILE_OUT)
def _split_get(data, fout1, fout2, value, frag):
    """Retrieve the split."""

    n = len(data)

    data = read_stage_file(data)
    if n > 0:
        data = data.sample(frac=1).reset_index(drop=True)
        split2 = data.iloc[value:].reset_index(drop=True)
        data = data.iloc[:value].reset_index(drop=True)
    else:
        split2 = data.copy()

    info1 = generate_info(data, frag)
    info2 = generate_info(split2, frag)

    save_stage_file(fout1, data)
    save_stage_file(fout2, split2)

    return info1, info2
