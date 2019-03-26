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
    df[col1] = np.random.random_integers(0, size*10, size=size).tolist()
    return df


def generate_data(total_size, nfrag, col1):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(s, col1) for s in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- Sample --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    col1 = 'col_1'

    t1 = time.time()
    df_list = generate_data(total_size, nfrag, col1)

    ddf1 = DDF().import_data(df_list)

    t2 = time.time()

    ddf1 = ddf1.sample(0.2, seed=123).cache()

    t3 = time.time()

    print "t2-t1:", t2 - t1
    print "t3-t2:", t3 - t2

    print "t_all:", t3 - t1
    # print ddf1.toDF()
