#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF, generate_info

import pandas as pd
import numpy as np
import time
import sys


@task(returns=2)
def generate_partition(size, col1, col2):
    df = pd.DataFrame()

    np.random.seed(123)
    df[col1] = np.random.normal(0, 0.1, size)
    df[col2] = df[col1] + 5

    info = generate_info(df)

    return df, info


def generate_data(total_size, nfrag, col1, col2):

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col1, col2)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col1 = 'col1'
    col2 = 'col2'

    t1 = time.time()
    df_list, info_schema = generate_data(n_rows, n_frag, col1, col2)
    ddf1 = DDF().import_data(df_list, info_schema)
    t2 = time.time()

    value = ddf1.correlation(col1, col2)

    print("Pearson correlation: ", value)

    t3 = time.time()

    print("t_all:", t3 - t1)
    print("t2-t1:", t2 - t1)
    print("t3-t2:", t3 - t2)
