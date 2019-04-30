#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_barrier
from pycompss.api.task import task
from ddf_library.ddf import DDF, generate_info

import pandas as pd
import numpy as np
import time
import sys


@task(returns=2)
def generate_partition(size, col_feature, col_label):
    df = pd.DataFrame({col_feature: np.random.normal(size=size),
                       col_label: np.random.randint(0, 10000, size=size)})

    info = generate_info(df)
    return df, info


def generate_data(total_size, nfrag, col_feature, col_label):
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_feature, col_label)

    return dfs, info


if __name__ == "__main__":

    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    col1 = 'col_1'
    col2 = 'col_2'

    t1 = time.time()
    df_list, info = generate_data(total_size, nfrag, col1, col2)
    ddf1 = DDF().import_data(df_list, info)
    t2 = time.time()

    ddf1 = ddf1.sample(0.2, seed=123).cache()
    compss_barrier()
    t3 = time.time()

    print("t2-t1:", t2 - t1)
    print("t3-t2:", t3 - t2)
    print("t_all:", t3 - t1)
    # print ddf1.to_df()
