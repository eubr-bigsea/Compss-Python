#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.api import compss_barrier
from ddf_library.ddf import DDF
from ddf_library.utils import generate_info

import pandas as pd
import numpy as np
import time
import sys


@task(returns=2)
def generate_partition(size, col_feature, col_label, frag):
    df = pd.DataFrame()
    df[col_feature] = np.random.normal(size=size)
    df[col_label] = np.random.random_integers(0, 10000, size=size)
    info = generate_info(df, frag)

    return df, info


def generate_data(total_size, nfrag, col1, col2):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col1, col2, f)

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

    ddf1 = ddf1.select([col1])\
        .filter("col_1 > 0.0")\
        .select_expression('col2 = col_1 * -1').cache()
    compss_barrier()
    t3 = time.time()

    print("t2-t1:", t2 - t1)
    print("t3-t2:", t3 - t2)

    print("t_all:", t3 - t1)
    # print len(ddf1.to_df())
