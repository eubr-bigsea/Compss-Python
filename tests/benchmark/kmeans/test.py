#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.api import compss_barrier

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info
from ddf_library.functions.ml.clustering import Kmeans

import pandas as pd
import numpy as np
import time


@task(returns=2)
def generate_partition(size, col_feature, dim, frag):
    df = pd.DataFrame(np.random.standard_normal(size=(size, dim)),
                      columns=col_feature)
    info = generate_info(df, frag)
    return df, info


def generate_data(total_size, nfrag, col_feature, dim):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_feature, dim, f)

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
    print("Time to generate and import data - t2-t1:", t2 - t1)

    ddf_train, ddf_test = ddf1.split(0.70)
    ddf_train = ddf_train.cache()
    ddf_test = ddf_test.cache()
    compss_barrier()
    t3 = time.time()
    print("Time to split data - t3-t2:", t3 - t2)

    sv = Kmeans(feature_col=cols, max_iters=20, n_clusters=2,
                init_mode='random', epsilon=0.01).fit(ddf_train)
    compss_barrier()
    t4 = time.time()

    print("Time to fit Kmeans - t4-t3:", t4 - t3)
    ddf1 = sv.transform(ddf1).cache()
    compss_barrier()
    t5 = time.time()
    print("Time to transform Kmeans - t5-t4:", t5 - t4)

    print("t_all:", t5 - t1)
    # ddf1.show()

