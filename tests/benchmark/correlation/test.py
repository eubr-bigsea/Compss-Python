#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF

import pandas as pd
import numpy as np
import time


@task(returns=1)
def generate_partition(size, col1, col2):
    df = pd.DataFrame()

    np.random.seed(123)
    v1 = np.random.normal(0, 0.1, size)
    df[col1] = v1
    df[col2] = v1 + 5

    return df


def generate_data(total_size, nfrag, col1, col2):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(s, col1, col2) for s in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- Pearson Correlation --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    col1 = 'col1'
    col2 = 'col2'

    t1 = time.time()
    df_list = generate_data(total_size, nfrag, col1, col2)

    ddf1 = DDF().import_data(df_list)

    t2 = time.time()

    value = ddf1.correlation(col1, col2)

    print "Pearson correlation: ", value

    t3 = time.time()

    print "t_all:", (t3-t1)
    print "t2-t1:", (t2 - t1)
    print "t3-t1:", (t3 - t2)

