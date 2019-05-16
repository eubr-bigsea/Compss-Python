#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info

from pycompss.api.api import compss_barrier
from pycompss.api.task import task

import pandas as pd
import numpy as np
import time


@task(returns=2)
def generate_partition(start, size, col1, col2, frag):
    x = np.arange(start, start+size, dtype=int)
    df = pd.DataFrame({col1: x, col2: x})

    info_schema = generate_info(df, frag)
    return df, info_schema


def generate_data(total_size, nfrag, col1, col2):
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    cumulative = 0
    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(cumulative, s, col1, col2, f)
        cumulative += s

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col1 = 'col_1'
    col2 = 'group'

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, col1, col2)
    ddf1 = DDF().import_data(df_list, info)
    compss_barrier()
    t2 = time.time()
    print("Time to generate and import data - t2-t1:", t2 - t1)

    ddf2 = ddf1.sample(0.5).cache()
    compss_barrier()
    t3 = time.time()
    print("Time to sample t3-t2:", t3 - t2)

    ddf1 = ddf1.intersect(ddf2).cache()
    compss_barrier()
    t4 = time.time()
    print("Time to intersect t4-t3:", t4 - t3)
    print("t_all:", t3 - t1)
    # ddf1.show()
    # print(ddf1.schema())
    print(ddf1.count_rows(total=False))
