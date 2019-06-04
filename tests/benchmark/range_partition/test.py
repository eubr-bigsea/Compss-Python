#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_barrier, compss_delete_object
from pycompss.api.task import task
from ddf_library.ddf import DDF
from ddf_library.utils import generate_info

import pandas as pd
import numpy as np
import time


@task(returns=2)
def generate_partition(size, col_1, col_2, frag):
    df = pd.DataFrame({col_1: np.random.normal(size=size),
                       col_2: np.random.randint(0, 10000*size, size=size)})
    info = generate_info(df, frag)
    return df, info


def generate_data(total_size, nfrag, col_1, col_2):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_1, col_2, f)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col_1 = 'col_1'
    col_2 = 'col_2'

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, col_1, col_2)
    ddf1 = DDF().import_data(df_list, info)
    compss_delete_object(df_list)
    t2 = time.time()
    print("Time to import data (t2-t1):", t2 - t1)

    ddf1 = ddf1.range_partition([col_1], [True], n_frag).cache()
    compss_barrier()
    t3 = time.time()
    print("t3-t2:", t3 - t2)
    print("t_all:", t3 - t1)

    print(ddf1.count_rows(total=False))
