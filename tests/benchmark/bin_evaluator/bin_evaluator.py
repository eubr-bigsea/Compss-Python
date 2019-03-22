#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF
from ddf_library.functions.ml.evaluation import BinaryClassificationMetrics

import pandas as pd
import numpy as np
import time


@task(returns=1)
def generate_partition(size, col_prd, col_label, _):
    df = pd.DataFrame()
    df[col_prd] = np.random.random_integers(0, 1, size=size).tolist()
    df[col_label] = np.random.random_integers(0, 1, size=size).tolist()
    return df


def generate_data(total_size, nfrag, col_feature, col_label, dim):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(s, col_feature, col_label, dim) for s in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- Binary Evaluator --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    dim = 2
    col_prd = 'prediction'
    col_label = 'label'

    t1 = time.time()
    df_list = generate_data(total_size, nfrag, col_prd, col_label, dim)

    ddf1 = DDF().import_data(df_list)

    t2 = time.time()

    bin_ev = BinaryClassificationMetrics(col_label, col_prd, ddf1)

    t3 = time.time()

    print bin_ev.get_metrics()
    print "\n"
    print bin_ev.confusion_matrix

    print "t2-t1:", t2 - t1
    print "t3-t2:", t3 - t2


