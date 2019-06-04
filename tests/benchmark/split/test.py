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
def generate_partition(size, cols, dim, frag):

    df = pd.DataFrame(np.random.randint(1, 1000*size, size=(size, dim)),
                      columns=cols)
    info = generate_info(df, frag)
    return df, info


def generate_data(total_size, nfrag, col_name, dim):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_name, dim, f)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    cols = ['column1', 'column2']

    t1 = time.time()
    print("Generating synthetic data (", n_rows, ") rows...")
    df_list, info = generate_data(n_rows, n_frag, cols, 2)
    ddf1 = DDF().import_data(df_list, info)

    compss_barrier()
    t2 = time.time()
    print("t2-t1:", t2 - t1)

    print("Running operation/algorithm...")
    ddf1a, ddf1b = ddf1.split(0.2)
    ddf1a.cache()
    ddf1b.cache()
    compss_barrier()
    t3 = time.time()
    print("Time to split (t3-t2):", t3 - t2)
    print("t_all:", t3 - t1)

    # print len(ddf1a.to_df())
    # print len(ddf1b.to_df())

    # print ddf1a.to_df()
    # print ddf1b.to_df()
