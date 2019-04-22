#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info
from ddf_library.functions.ml.feature import MinMaxScaler

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
    dim = 2
    cols = ["col{}".format(c) for c in range(dim)]

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, cols, dim)
    ddf1 = DDF().import_data(df_list, info)

    compss_barrier()
    t2 = time.time()
    print("Time to generate and to import_data (t2-t1):", t2 - t1)

    # ddf_train, ddf_test = ddf1.split(0.5)
    # ddf_train = ddf_train.cache()
    # ddf_test = ddf_test.cache()

    # compss_barrier()
    # t3 = time.time()
    # print("Time to split (t3-t2):", t3 - t2)

    scaler = MinMaxScaler(input_col=cols).fit(ddf1)
    t3 = time.time()
    print("Time to MinMaxScaler.fit (t3-t2):", t3 - t2)

    ddf1 = scaler.transform(ddf1)
    compss_barrier()
    t4 = time.time()
    print("Time to MinMaxScaler.transform (t4-t3):", t4 - t3)
    print("t_all:", t4 - t1)

    # print(ddf_train.count())
    # print(ddf_test.count())
