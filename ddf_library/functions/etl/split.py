#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.local import local
from pycompss.api.task import task
import numpy as np
import math


def split(data, settings):
    """
    Randomly splits a DataFrame into two distintics DataFrames.

    :param data: A list with nfrag pandas's DataFrame;
    :param settings: A dictionary that contains:
      - 'percentage': Percentage to split the data (default, 0.5);
      - 'seed': Optional, seed in case of deterministic random operation;
      - 'info': information generated from others tasks (automatic);
    :return: Returns two lists with nfrag pandas's DataFrame with
     distincts subsets of the input.

    ..note: The operation consists of two stages: first, we define the
     distribution; and the second we split the data. The first part cannot be
     grouped with others stages because we need information about the
     size in each partition. The second part cannot be grouped as well because
     generates two differents outputs data.
    """
    nfrag = len(data)

    idxs, seed = _preprocessing(settings, nfrag)
    out1 = [[] for _ in range(nfrag)]
    out2 = [[] for _ in range(nfrag)]
    info1 = [[] for _ in range(nfrag)]
    info2 = [[] for _ in range(nfrag)]

    for i, fraction in enumerate(idxs):
        out1[i], out2[i], info1[i], info2[i] = _split_get(data[i],
                                                          fraction, seed)

    output = {'key_data': ['data1', 'data2'],
              'key_info': ['info1', 'info2'],
              'data1': out1, 'info1': info1,
              'data2': out2, 'info2': info2}
    return output


def _preprocessing(settings, nfrag):
    percentage = settings.get('percentage', 0.5)
    seed = settings.get('seed', None)
    info = settings['info'][0]

    if percentage < 0 or percentage > 1:
        raise Exception("Please inform a valid percentage [0, 1].")

    idxs = _split_allocate(info, percentage, seed, nfrag)
    return idxs, seed


@local
def _split_allocate(info, percentage, seed, nfrag):
    """Define a list of indexes to be splitted."""

    n_list = info[2]

    total = sum(n_list)

    size = int(math.ceil(total*percentage))

    np.random.seed(seed)

    sizes = [int(math.ceil(n * percentage)) for n in n_list]

    val = sum(sizes)
    for i in range(nfrag):
        if val == size:
            break
        if sizes[i] > 0:
            sizes[i] -= 1
            val -= 1

    return sizes


@task(returns=4)
def _split_get(data, value, seed):
    """Retrieve the split."""
    n = len(data)

    if n > 0:
        data.reset_index(drop=True, inplace=True)
        split1 = data.sample(n=value, replace=False, random_state=seed)
        data = data.drop(split1.index)

        split1.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

    else:
        split1 = data.copy()

    info1 = [split1.columns.tolist(), split1.dtypes.values, [len(split1)]]
    info2 = [data.columns.tolist(), data.dtypes.values, [len(data)]]

    return split1, data, info1, info2
