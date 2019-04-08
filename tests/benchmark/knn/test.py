#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_barrier
from pycompss.api.task import task
from ddf_library.ddf import DDF, generate_info
from ddf_library.functions.ml.classification import KNearestNeighbors

import pandas as pd
import numpy as np


@task(returns=2)
def generate_partition(size, col_feature, col_label, dim):
    df = pd.DataFrame({col_feature: np.random.normal(size=(size, dim)),
                       col_label: np.random.random_integers(1, 5, size=size)})

    info = generate_info(df)
    return df, info


def generate_data(total_size, nfrag, col_feature, col_label, dim):

    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_feature, col_label, dim)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    dim = 2
    col_feature = 'features'
    col_label = 'label'

    df_list, info = generate_data(n_rows, n_frag, col_feature, col_label, dim)

    ddf_train, ddf_test = DDF().import_data(df_list, info).split(0.10, seed=123)

    kNN = KNearestNeighbors(feature_col=col_feature,
                            label_col=col_label, k=1).fit(ddf_train)

    ddf_test = kNN.transform(ddf_test).cache()
    compss_barrier()
    # ddf_test.show()


