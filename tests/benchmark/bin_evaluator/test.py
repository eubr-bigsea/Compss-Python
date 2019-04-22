#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info
from ddf_library.functions.ml.evaluation import BinaryClassificationMetrics

from pycompss.api.task import task
from pycompss.api.api import compss_barrier

import pandas as pd
import numpy as np
import time
import sys


@task(returns=2)
def generate_partition(size, col_prd, col_label, frag):
    df = pd.DataFrame({col_prd: np.random.randint(0, 2, size=size),
                       col_label: np.random.randint(0, 2, size=size)})

    info = generate_info(df, frag)
    return df, info


def generate_data(total_size, nfrag, col_feature, col_label):
    dfs = [[] for _ in range(nfrag)]
    info = [[] for _ in range(nfrag)]

    size = total_size // nfrag
    sizes = [size for _ in range(nfrag)]
    sizes[-1] += (total_size - sum(sizes))

    for f, s in enumerate(sizes):
        dfs[f], info[f] = generate_partition(s, col_feature, col_label, f)

    return dfs, info


if __name__ == "__main__":

    n_rows = int(sys.argv[1])
    n_frag = int(sys.argv[2])
    col_prd = 'prediction'
    col_label = 'label'

    t1 = time.time()
    df_list, info = generate_data(n_rows, n_frag, col_prd, col_label)
    ddf1 = DDF().import_data(df_list, info)
    compss_barrier()
    t2 = time.time()

    bin_ev = BinaryClassificationMetrics(col_label, col_prd, ddf1)
    compss_barrier()
    t3 = time.time()

    print(bin_ev.get_metrics(), "\n\n", bin_ev.confusion_matrix)

    print("Time to generate and import data - t2-t1:", t2 - t1)
    print("Time to BinaryClassificationMetrics t3-t2:", t3 - t2)
    print("t_all:", t3 - t1)


