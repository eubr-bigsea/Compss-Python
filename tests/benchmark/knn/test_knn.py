#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF
from ddf_library.functions.ml.classification import KNearestNeighbors

import pandas as pd
import numpy as np


@task(returns=1)
def generate_partition(size, col_feature, col_label, dim):
    df = pd.DataFrame()
    df[col_feature] = np.random.normal(size=(size, dim)).tolist()
    df[col_label] = np.random.random_integers(1, 5, size=size).tolist()
    return df


def generate_data(total_size, nfrag, col_feature, col_label, dim):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(s, col_feature, col_label, dim) for s in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- kNN --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    dim = 2
    col_feature = 'features'
    col_label = 'label'

    df_list = generate_data(total_size, nfrag, col_feature, col_label, dim)

    ddf_train, ddf_test = DDF().import_data(df_list).split(0.10, seed=123)

    kNN = KNearestNeighbors(feature_col=col_feature,
                            label_col=col_label, k=1).fit(ddf_train)

    ddf_test = kNN.transform(ddf_test).cache()
    # print ddf_test.show()
