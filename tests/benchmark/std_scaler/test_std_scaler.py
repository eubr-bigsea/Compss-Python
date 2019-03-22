#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF
from ddf_library.functions.ml.feature import VectorAssembler, StandardScaler

import pandas as pd
import numpy as np


@task(returns=1)
def generate_partition(size, col_name, dim):
    df = pd.DataFrame()
    df[col_name] = np.random.random_integers(1, 1000, size=(size, dim)).tolist()
    return df


def generate_data(total_size, nfrag, col_name, dim):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(size, col_name, dim) for size in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- STD Scaler --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    dim = 2
    col_name = 'features'

    df_list = generate_data(total_size, nfrag, col_name, dim)

    ddf1 = DDF().import_data(df_list)

    ddf1 = StandardScaler(input_col=col_name).fit_transform(ddf1).cache()
    # print ddf1.show()
