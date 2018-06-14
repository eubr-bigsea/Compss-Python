#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np

class MaxAbsScalerOperation(object):
    """MaxAbs Scaler Operation.

    MaxAbsScaler transforms a dataset of features rows,
    rescaling each feature to range [-1, 1] by dividing through
    the maximum absolute value in each feature.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the
    training set will be 1.0. It does not shift/center the data,
    and thus does not destroy any sparsity.
    """

    def fit(self, data, settings, nfrag):

        columns = settings.get('attributes', [])

        if len(columns) == 0:
            raise Exception('You must insert the fields to '
                            'perform the Normalization.')

        # generate a list of the min and the max element to each subset.
        minmax_partial = \
            [_agg_maxabs(data[f], columns) for f in range(nfrag)]

        # merge them into only one list
        minmax = mergeReduce(_merge_maxabs, minmax_partial)
        return minmax

    def transform(self, data, model, settings, nfrag):
        """MaxAbs Scaler Operation.

        :param data: A list with nfrag pandas's dataframe to perform
            the Normalization.
        :param settings: A dictionary that contains:
          - attributes: A list of features names to normalize;
          - alias: Aliases of the new columns (overwrite the fields if empty);
        :param nfrag: A number of fragments;
        :return: A list with nfrag pandas's dataframe
        """
        settings['attributes'] = settings.get('attributes', [])

        if len(settings['attributes']) == 0:
            raise Exception('You must insert the fields to '
                            'perform the Normalization.')

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _minmax_scaler(data[f], model, settings)

        return result

    def transform_serial(self, data, model, settings):
        return _minmax_scaler_(data, model, settings)


@task(returns=list)
def _agg_maxabs(df, columns):
    """Generate a list of min and max values, excluding NaN values."""
    min_max_p = []
    for col in columns:
        p = [np.min(df[col].values, axis=0), np.max(df[col].values, axis=0)]
        min_max_p.append(p)
    return min_max_p


@task(returns=list)
def _merge_maxabs(minmax1, minmax2):
    """Merge max abs values."""
    maxabs = []
    for feature in zip(minmax1, minmax2):
        di, dj = feature
        minimum = di[0] if di[0] < dj[0] else dj[0]
        maximum = di[1] if di[1] > dj[1] else dj[1]
        maxabs.append([minimum, maximum])
    return maxabs


@task(returns=list)
def _minmax_scaler(data, minmax, settings):
    return _minmax_scaler_(data, minmax, settings)


def _minmax_scaler_(data, minmax, settings):
    """Normalize by range mode."""
    features = settings['attributes']
    alias = settings.get('alias', [])

    if len(alias) != len(features):
        alias = features

    for i, (alias, col) in enumerate(zip(alias, features)):
        minimum, maximum = minmax[i]
        data[alias] = data[col].apply(
                lambda xs: calculation(xs, minimum, maximum))

    return data


def calculation(xs, minimum, maximum):
    features = []
    for xi, mi, ma in zip(xs, minimum, maximum):
        ma = abs(ma)
        mi = abs(mi)
        maxabs = ma if ma > mi else mi
        v = float(xi) / maxabs
        features.append(v)
    return features
