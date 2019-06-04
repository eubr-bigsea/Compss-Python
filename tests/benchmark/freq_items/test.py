#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF, generate_info

import pandas as pd
import numpy as np
import time
import sys


@task(returns=2)
def generate_partition(size, col_name):
    df = pd.DataFrame(np.random.randint(1, 10000, size=size),
                      columns=[col_name])

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


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col_name = 'feature'

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, col_name)
    ddf1 = DDF().import_data(df_list, info)
    t2 = time.time()

    df = ddf1.freq_items([col_name], support=0.2)
    print(df)
    t3 = time.time()

    print("t2-t1:", t2 - t1)
    print("t3-t2:", t3 - t2)
    print("t_all:", t3 - t1)
