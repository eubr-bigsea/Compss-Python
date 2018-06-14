#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np


class IndexToStringOperation(object):
    """Feature Indexer.

    Symmetrically to StringIndexer, IndexToString maps a column of
    label indices back to a column containing the original labels as strings.
    """
    def transform(self, data, settings, nfrag):
        """
        IndexToStringOperation.

        :param data: A list with nfrag pandas's dataFrame;
        :param settings: A dictionary that contains:
            - 'inputCol': Field to perform the operation,
            - 'outputCol': Alias to the converted field (default, add a
                    suffix '_indexed');
            - 'model': A model created by the StringIndexerOperation;
        :param nfrag: A number of fragments;
        :return  Returns a new dataFrame with the indexed field.
        """

        settings = self.preprocessing(settings)
        inputCol = settings['inputCol']
        outputCol = settings.get('outputCol', "{}_indexed")

        model = settings['model']['model']
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _index_to_string(data[f], inputCol, outputCol, model)
        return result

    def preprocessing(self, settings):
        """Validation step and inital settings."""
        if 'inputCol' not in settings:
            raise Exception("You must inform the `inputCol` field.")

        mode = settings.get('IndexToString', False)

        if mode:
            if 'model' not in settings:
                raise Exception("You must inform the `model` setting.")
            if settings['model'].get('algorithm', '') \
                    != 'FeatureIndexerOperation':
                raise Exception("You must inform the valid `model`.")


@task(returns=list)
def get_indexes(data, inputCol):
    """Create partial model to convert string to index."""
    x = data[inputCol].dropna().unique()
    return x


@task(returns=list)
def merge_mapper(data1, data2):
    """Merge partial models into one."""
    data1 = np.concatenate((data1, data2), axis=0)
    return np.unique(data1)


@task(returns=list)
def _string_to_indexer(data, inputCol, outputCol, mapper):
    """Convert string to index based in the model."""
    news = [i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[outputCol] = data[inputCol].replace(to_replace=mapper, value=news)
    return data


@task(returns=list)
def _index_to_string(data, inputCol, outputCol, mapper):
    """Convert index to string based in the model."""
    news = [i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[outputCol] = data[inputCol].replace(to_replace=news, value=mapper)
    return data
