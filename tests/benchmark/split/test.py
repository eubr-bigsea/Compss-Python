#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from pycompss.api.api import compss_barrier
from pycompss.api.task import task
from ddf_library.ddf import DDF, generate_info

import pandas as pd
import numpy as np
import time
import sys



@task(returns=2)
def generate_partition(size, col1, col2):
    df = pd.DataFrame({col1: np.random.randint(0, size*100, size=size),
                       col2: np.random.randint(0, size*100, size=size)})

    info = generate_info(df)
    return df, info


def generate_data(total_size, n_frag, col1, col2):
    dfs = [[] for _ in range(n_frag)]
    info = [[] for _ in range(n_frag)]

    size = total_size // n_frag
    sizes = [size for _ in range(n_frag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col1, col2)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col1 = 'column1'
    col2 = 'column2'

    t1 = time.time()
    print("Generating synthetic data (", n_rows, ") rows...")
    df_list, info = generate_data(n_rows, n_frag, col1, col2)
    ddf1 = DDF().import_data(df_list, info)

    t2 = time.time()
    print("Running operation/algorithm...")
    ddf1a, ddf1b = ddf1.split(0.2, seed=None)
    ddf1a.cache()
    ddf1b.cache()
    # compss_barrier()
    t3 = time.time()

    print("t2-t1:", t2 - t1)
    print("t3-t2:", t3 - t2)
    print("t_all:", t3 - t1)

    # print len(ddf1a.to_df())
    # print len(ddf1b.to_df())

    # print ddf1a.to_df()
    # print ddf1b.to_df()
