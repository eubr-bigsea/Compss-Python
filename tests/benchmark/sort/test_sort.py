#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from ddf_library.ddf import DDF
import pandas as pd
import numpy as np


@task(returns=1)
def generate_partition(size, name):
    df = pd.DataFrame()
    df[name] = np.random.normal(size=size)
    return df


def generate_data(total_size, nfrag, col_name):

    size = total_size / nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    dfs = [generate_partition(size, col_name) for size in sizes]

    return dfs


if __name__ == "__main__":
    print "\n|-------- Sort --------|\n"
    total_size = int(sys.argv[1])
    nfrag = int(sys.argv[2])
    col_name = 'col_0'
    df_list = generate_data(total_size, nfrag, col_name)

    ddf1 = DDF().import_data(df_list).sort([col_name], ascending=[True]).cache()

    # print ddf1.show()