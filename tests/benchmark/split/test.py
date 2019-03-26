#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF

import pandas as pd
import numpy as np
import time


@task(returns=1)
def generate_partition(size, col1):
    df = pd.DataFrame()

    np.random.seed(None)
    df[col1] = np.random.randint(0, size*100, size=size).tolist()
    return df


def generate_data(total_size, nfrag, col1):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(s, col1) for s in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- Split --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    col1 = 'col_1'

    t1 = time.time()
    df_list = generate_data(total_size, nfrag, col1)
    ddf1 = DDF().import_data(df_list)

    # df = pd.DataFrame()
    # df[col1] = np.arange(total_size).tolist()
    # ddf1 = DDF().parallelize(df, nfrag)

    t2 = time.time()

    ddf1a, ddf1b = ddf1.split(0.2, seed=None)

    ddf1a.cache()

    ddf1b.cache()

    t3 = time.time()

    print "t2-t1:", t2 - t1
    print "t3-t2:", t3 - t2

    print "t_all:", t3 - t1

    # print len(ddf1a.toDF())
    # print len(ddf1b.toDF())

    # print ddf1a.toDF()
