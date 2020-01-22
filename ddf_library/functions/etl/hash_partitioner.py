#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info, save_stage_file, create_stage_files

from pycompss.api.parameter import FILE_OUT, COLLECTION_IN
from pycompss.api.task import task
from pycompss.api.api import compss_delete_object

import pandas as pd
import importlib


def hash_partition(data, settings):
    """
    A Partitioner that generates partitions organized by hash function.

    :param data: A list of pandas DataFrames;
    :param settings: A dictionary with:
     - info:
     - columns: Columns name to perform a hash partition;
     - nfrag: Number of partitions;

    :return: A list of pandas DataFrames;
    """
    info = settings['info'][0]
    nfrag = len(data)
    cols = settings['columns'] if settings['columns'] else []

    if len(cols) == 0:
        cols = info['cols']
        settings['columns'] = cols

    nfrag_target = settings.get('nfrag', nfrag)

    if nfrag_target < 1:
        raise Exception('You must have at least one partition.')

    elif nfrag_target > 1:

        import ddf_library.bases.config
        ddf_library.bases.config.x = nfrag_target

        splits = [[0 for _ in range(nfrag_target)] for _ in range(nfrag)]

        import ddf_library.functions.etl.repartition
        importlib.reload(ddf_library.functions.etl.repartition)

        for f in range(nfrag):
            splits[f] = ddf_library.functions.etl.repartition\
                .split_by_hash(data[f], cols, info, nfrag_target)

        # n_concat = nfrag // 2 if nfrag > 10 else nfrag
        result = create_stage_files(nfrag_target)
        info = [{} for _ in range(nfrag_target)]
        for f in range(nfrag_target):
            tmp = [splits[t][f] for t in range(nfrag)]
            info[f] = concat_n_pandas(result[f], f, tmp)

        compss_delete_object(splits)
    else:
        result = data

    output = {'key_data': ['data'], 'key_info': ['info'],
              'data': result, 'info': info}
    return output


# @constraint(ComputingUnits="2")  # approach to have more memory
@task(data_out=FILE_OUT, args=COLLECTION_IN, returns=1)
def concat_n_pandas(data_out, f, args):
    dfs = [df for df in args if isinstance(df, pd.DataFrame)]
    dfs = pd.concat(dfs, ignore_index=True, sort=False)
    del args
    dfs = dfs.infer_objects()
    info = generate_info(dfs, f)
    save_stage_file(data_out, dfs)
    return info

