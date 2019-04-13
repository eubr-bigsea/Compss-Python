#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info
from ddf_library.functions.ml.feature import StandardScaler, VectorAssembler

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
    col_name = 'features'

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, cols, dim)
    ddf1 = DDF().import_data(df_list, info)
    ddf1 = VectorAssembler(input_col=cols, output_col=col_name, remove=True)\
        .transform(ddf1).cache()

    compss_barrier()
    t2 = time.time()
    print("t2-t1:", t2 - t1)

    ddf_train, ddf_test = ddf1.split(0.7)
    ddf_train = ddf_train.cache()
    ddf_test = ddf_test.cache()

    compss_barrier()
    t3 = time.time()
    print("t3-t1:", t3 - t1)

    scaler = StandardScaler(input_col=col_name).fit(ddf_train)
    ddf1 = scaler.transform(ddf1)
    compss_barrier()
    t4 = time.time()
    print("t4-t3:", t4 - t3)
    print("t_all:", t4 - t1)

    # print(ddf_train.count())
    # print(ddf_test.count())
