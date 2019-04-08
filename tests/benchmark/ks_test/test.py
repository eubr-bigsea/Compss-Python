#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF, generate_info

import pandas as pd
import numpy as np


@task(returns=2)
def generate_partition(size, col_name):
    np.random.seed(123)
    df = pd.DataFrame({col_name: np.random.normal(1, 1000, size=size)})
    info = generate_info(df)
    return df, info


def generate_data(total_size, nfrag, col_name):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_name)

    return dfs, info


def local(size):
    from scipy import stats
    np.random.seed(123)
    x = np.random.normal(1, 1000, size=size).tolist()
    print(stats.kstest(x, 'norm'))


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col_name = 'feature'

    df_list, info = generate_data(n_rows, n_frag, col_name)

    result = DDF().import_data(df_list, info).kolmogorov_smirnov_one_sample(col_name)
    print(result)
