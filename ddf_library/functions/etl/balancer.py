#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_INOUT
from .parallelize import _generate_distribution2
from ddf_library.utils import generate_info,\
    create_stage_files, read_stage_file, save_stage_file

import numpy as np
import pandas as pd
import time


class WorkloadBalancer(object):
    """
    Redistribute the data in equal parts if it's unbalanced.
    It is considered an unbalanced DataFrame if the coefficient of
    variation (CV) between fragments is greater than 0.20.

    This method keeps the number of fragments. Use repartition if you want
    change it.
    """

    def __init__(self, settings):

        forced = settings['forced']

        self.info = settings['schema'][0]
        sizes = self.info['size']
        self.is_balanced = self.preprocessing(sizes, forced, len(sizes))

    def preprocessing(self, sizes, forced, nfrag):
        """Check the distribution of the data."""

        if nfrag != len(sizes):
            return False

        if forced:
            return self.check_is_strong_balanced(sizes)

        else:
            cv = np.std(sizes) / np.mean(sizes)
            print("Coefficient of variation:{}".format(cv))
            if cv > 0.20:
                return False
            else:
                return True

    @staticmethod
    def check_is_strong_balanced(size):
        min_f, max_f = min(size), max(size)
        if max_f - min_f <= 1:
            return True
        else:
            return False

    def transform(self, data):
        """
        :param data: A list with nfrag pandas's DataFrame;
        :return: Returns a balanced list with nfrag pandas's DataFrame.
        """

        info = self.info

        cols = info['cols']
        if self.is_balanced:
            print("Data is already balanced.")
            result = data

        else:
            old_sizes = info['size']
            nfrag = len(old_sizes)
            n_rows = sum(old_sizes)
            target_sizes = _generate_distribution2(n_rows, nfrag)
            result, info = _balancer(data, target_sizes, old_sizes, cols)

        output = {'key_data': ['data'], 'key_info': ['schema'],
                  'data': result, 'schema': info}

        return output


def _balancer(data, target_sizes, old_sizes, cols):
    nfrag_new = len(target_sizes)
    nfrag_old = len(old_sizes)

    result = create_stage_files(nfrag_new)
    for r in result:
        pd.DataFrame(columns=cols).to_parquet(r)
    info = [[] for _ in range(nfrag_new)]

    matrix = np.zeros((nfrag_new, nfrag_old), dtype=int)
    print('Repartition fragments to:', target_sizes)

    for f in range(nfrag_new):
        target = target_sizes[f]
        for f2 in range(nfrag_old):
            actual = matrix[f].sum()
            diff = target - actual
            if diff > 0:
                nb = old_sizes[f2]
                if nb > 0:
                    if nb > diff:
                        matrix[f, f2] = diff
                        old_sizes[f2] -= diff
                    else:
                        matrix[f, f2] = nb
                        old_sizes[f2] -= nb

    for f in range(nfrag_new):
        for f2 in range(nfrag_old):
            size = int(matrix[f, f2])
            if size > 0:
                head = int(np.sum(matrix[:, f2][0:f]))
                info[f] = _balancer_get_rows(result[f], data[f2],
                                             head, size, f)

    return result, info


@task(returns=1, result=FILE_INOUT, data=FILE_IN)
def _balancer_get_rows(result, data, head, size, f):
    t_start = time.time()

    df = read_stage_file(result)
    data = read_stage_file(data)
    data.reset_index(drop=True, inplace=True)
    portion = data.iloc[head: head+size]
    del data

    if len(df) == 0:
        df = portion
    else:
        df = pd.concat([df, portion], sort=False, ignore_index=True)
        df = df.infer_objects()
        # TODO: workaround
        for col in df.columns:
            coltype = str(df[col].dtype)
            if coltype == 'object':
                df[col] = df[col].astype(str)

    info = generate_info(df, f)
    save_stage_file(result, df)

    t_end = time.time()
    print("[INFO] - Time to process task '{}': {:.0f} seconds"
          .format('_balancer_get_rows', t_end - t_start))
    return info
