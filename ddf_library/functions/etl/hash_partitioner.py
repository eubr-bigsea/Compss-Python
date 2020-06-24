#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import create_stage_files
from ddf_library.bases.tasks import concat_n_pandas

from pycompss.api.api import compss_delete_object


import importlib


def hash_partition(data, settings):
    """
    A Partitioner that generates partitions organized by hash function.

    :param data: A list of pandas DataFrames;
    :param settings: A dictionary with:
     - schema:
     - columns: Columns name to perform a hash partition;
     - nfrag: Number of partitions;

    :return: A list of pandas DataFrames;
    """
    info = settings['schema'][0]
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

        result = create_stage_files(nfrag_target)
        info = [{} for _ in range(nfrag_target)]
        for f in range(nfrag_target):
            tmp = [splits[t][f] for t in range(nfrag)]
            info[f] = concat_n_pandas(result[f], f, tmp)

        compss_delete_object(splits)
    else:
        # TODO: intermediate_result ou pode ignorar?
        result = data

    output = {'key_data': ['data'], 'key_info': ['schema'],
              'data': result, 'schema': info}
    return output



