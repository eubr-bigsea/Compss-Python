#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np


class StringIndexerOperation(object):
    """String Indexer.

    Indexes a feature by encoding a string column as a
    column containing indexes.
    """
    def transform(self, data, settings, nfrag):
        """
        :param data: A list with nfrag pandas's dataframe;
        :param settings: A dictionary that contains:
            - 'inputCol': Field to perform the operation,
            - 'outputCol': Alias to the converted field (default, add a
                    suffix '_indexed');
        :param nfrag: A number of fragments;
        :return  Returns a new dataframe with the indexed field.
        """

        in_col, out_col, mapper = self.preprocessing(settings, data, nfrag)
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _string_to_indexer(data[f], in_col, out_col, mapper)

        model = dict()
        model['algorithm'] = 'FeatureIndexerOperation'
        model['model'] = mapper
        return [result, model]

    def preprocessing(self, settings, data, nfrag):
        """Validation step and inital settings."""
        if 'inputCol' not in settings:
            raise Exception("You must inform the `inputCol` field.")

        in_col = settings['inputCol']
        out_col = settings.get('outputCol', "{}_indexed")

        mapper = [get_indexes(data[f], in_col) for f in range(nfrag)]
        mapper = mergeReduce(merge_mapper, mapper)

        return in_col, out_col, mapper


@task(returns=list)
def get_indexes(data, in_col):
    """Create partial model to convert string to index."""
    x = data[in_col].dropna().unique()
    return x


@task(returns=list)
def merge_mapper(data1, data2):
    """Merge partial models into one."""
    data1 = np.concatenate((data1, data2), axis=0)
    return np.unique(data1)


@task(returns=list)
def _string_to_indexer(data, in_col, out_col, mapper):
    """Convert string to index based in the model."""
    news = [i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[out_col] = data[in_col].replace(to_replace=mapper, value=news)
    return data
