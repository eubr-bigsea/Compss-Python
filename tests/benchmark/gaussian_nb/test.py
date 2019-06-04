#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.api.api import compss_barrier

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info
from ddf_library.functions.ml.classification import GaussianNB

import pandas as pd
import numpy as np
import time


@task(returns=2)
def generate_partition(size, col_feature, col_label, dim, frag):
    df = pd.DataFrame(np.random.normal(size=(size, dim)), columns=col_feature)
    df[col_label] = np.random.random_integers(0, 2, size=size)

    info = generate_info(df, frag)
    return df, info


def generate_data(total_size, nfrag, col_feature, col_label, dim):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_feature, col_label, dim, f)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    dim = 2
    cols = ["col{}".format(c) for c in range(dim)]
    col_label = 'label'

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, cols, col_label, dim)
    ddf1 = DDF().import_data(df_list, info)
    compss_barrier()
    t2 = time.time()
    print("Time to generate and import data - t2-t1:", t2 - t1)

    ddf_train, ddf_test = ddf1.split(0.70, seed=123)
    ddf_train = ddf_train.cache()
    ddf_test = ddf_test.cache()
    compss_barrier()
    t3 = time.time()
    print("Time to split data - t3-t2:", t3 - t2)

    nb = GaussianNB(feature_col=cols, label_col=col_label).fit(ddf_train)
    compss_barrier()
    t4 = time.time()

    print("Time to fit GaussianNB - t4-t3:", t4 - t3)
    ddf_test = nb.transform(ddf_test).cache()
    compss_barrier()
    t5 = time.time()
    print("Time to transform GaussianNB - t5-t4:", t5 - t4)

    print("t_all:", t5 - t1)
    # ddf1.show()
