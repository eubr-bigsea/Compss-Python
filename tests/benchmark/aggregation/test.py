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
def generate_partition(size, col_feature, col_label, frag):
    df = pd.DataFrame({col_feature: np.random.randint(0, 100000, size=size),
                       col_label: np.random.randint(0, 100000, size=size)})

    info = generate_info(df, frag)
    return df, info


def generate_data(total_size, nfrag, col_feature, col_label):
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_feature, col_label, f)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col1 = 'col_1'
    col_label = 'group'

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, col1, col_label)
    ddf1 = DDF().import_data(df_list, info)
    compss_barrier()
    t2 = time.time()
    print("Time to generate and import data - t2-t1:", t2 - t1)

    ddf1 = ddf1.group_by([col1])\
        .first([col1])\
        .count([col1], alias=['Counter'])\
        .cache()
    compss_barrier()
    t3 = time.time()
    print("Time to aggragate t3-t2:", t3 - t2)
    print("t_all:", t3 - t1)
    # ddf1.show()
