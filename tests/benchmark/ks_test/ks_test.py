#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF

import pandas as pd
import numpy as np


@task(returns=1)
def generate_partition(size, col_name):
    df = pd.DataFrame()
    np.random.seed(123)
    df[col_name] = np.random.normal(1, 1000, size=size).tolist()
    return df


def generate_data(total_size, nfrag, col_name):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(size, col_name) for size in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- KS Test --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    col_name = 'feature'

    df_list = generate_data(total_size, nfrag, col_name)

    ddf1 = DDF().import_data(df_list).kolmogorov_smirnov_one_sample(col_name)
    print ddf1
