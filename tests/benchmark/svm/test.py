#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF
from ddf_library.functions.ml.classification import SVM

import pandas as pd
import numpy as np
import time


@task(returns=1)
def generate_partition(size, col_feature, col_label, dim):
    df = pd.DataFrame()
    df[col_feature] = np.random.normal(size=(size, dim)).tolist()
    df[col_label] = np.random.choice([-1, 1], size=size).tolist()
    return df


def generate_data(total_size, nfrag, col_feature, col_label, dim):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(s, col_feature, col_label, dim) for s in sizes]

    return dfs


# def plot(df_list, col_feature, col_label):
#
#     df = pd.concat(df_list)
#     xy = np.array(df[col_feature].tolist())
#     label = df[col_label].values
#
#     import matplotlib.pyplot as plt
#     plt.scatter(xy[:, 0], xy[:, 1], c=label)
#     plt.savefig('svm.png')


if __name__ == "__main__":
    print "\n|-------- SVM --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    dim = 2
    col_feature = 'features'
    col_label = 'label'

    t1 = time.time()

    df_list = generate_data(total_size, nfrag, col_feature, col_label, dim)

    ddf1 = DDF().import_data(df_list)

    t2 = time.time()

    ddf_train, ddf_test = ddf1.split(0.10, seed=123)

    t3 = time.time()

    sv = SVM(feature_col=col_feature,
             label_col=col_label, max_iters=20, coef_lr=0.1).fit(ddf_train)

    t4 = time.time()

    ddf_test = sv.transform(ddf_test).cache()

    t5 = time.time()

    print "t2-t1:", t2 - t1
    print "t3-t2:", t3 - t2
    print "t4-t3:", t4 - t3
    print "t5-t4:", t5 - t4
    print "t_all:", t5 - t1
    # print ddf_test.show()
    # plot(df_list, col_feature, col_label)
