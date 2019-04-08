#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_barrier
from pycompss.api.task import task
from ddf_library.ddf import DDF, generate_info

import pandas as pd
import numpy as np
import time


@task(returns=2)
def generate_partition(size, col_1, col_2):
    df = pd.DataFrame({col_2: np.random.normal(size=size),
                       col_1: np.random.randint(0, size*100, size=size)})
    info = generate_info(df)
    return df, info


def generate_data(total_size, nfrag, col_1, col_2):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_1, col_2)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col_1 = 'col_1'
    col_2 = 'col_2'

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, col_1, col_2)
    ddf1 = DDF().import_data(df_list, info)

    t2 = time.time()

    ddf1 = ddf1.sort([col_1], ascending=[True]).cache()
    compss_barrier()
    t3 = time.time()

    print("t2-t1:", t2 - t1)
    print("t3-t2:", t3 - t2)
    print("t_all:", t3 - t1)

    # print(ddf1.schema())
    # ddf1.show(100)
    c = ddf1.to_df()[col_1].values
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    print(is_sorted(c))
    print(len(c))
