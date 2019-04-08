#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF

import pandas as pd
import numpy as np
import time


@task(returns=1)
def generate_partition(size, col_feature, col_label):
    df = pd.DataFrame()
    df[col_feature] = np.random.normal(size=size).tolist()
    df[col_label] = np.random.random_integers(0, 10000, size=size).tolist()
    return df


def generate_data(total_size, nfrag, col_feature, col_label, dim):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(s, col_feature, col_label, dim) for s in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- Example Flow --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    dim = 2
    col1 = 'col_1'
    col_label = 'group'

    t1 = time.time()
    df_list = generate_data(total_size, nfrag, col1, col_label, dim)

    ddf1 = DDF().import_data(df_list)

    t2 = time.time()
    ddf1 = ddf1.select([col1])\
        .filter("col_1 > 0.0")\
        .map(lambda row: row['col_1'] + 1, 'col_2').cache()

    t3 = time.time()

    print "t2-t1:", t2 - t1
    print "t3-t2:", t3 - t2

    print "t_all:", t3 - t1
    # print len(ddf1.toDF())
