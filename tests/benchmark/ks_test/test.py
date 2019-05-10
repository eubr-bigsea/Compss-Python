#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.api import compss_barrier

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info

import pandas as pd
import numpy as np
import time


@task(returns=2)
def generate_partition(size, col_name, frag):
    df = pd.DataFrame({col_name: np.random.normal(1, size*10000, size=size)})
    info = generate_info(df, frag)
    return df, info


def generate_data(total_size, nfrag, col):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col, f)

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

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, col_name)
    ddf1 = DDF().import_data(df_list, info)
    compss_barrier()
    t2 = time.time()
    print("Time to generate data t2-t1:", t2 - t1)

    result = ddf1.kolmogorov_smirnov_one_sample(col_name)
    print(result)
    t3 = time.time()
    print("Time to test t3-t2:", t3 - t2)
    print("t_all:", t3 - t1)
