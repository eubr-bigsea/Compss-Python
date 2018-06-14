#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np


class MinMaxScalerOperation(object):
    """MinMax Scaler Operation.

    MinMaxScaler transforms a dataset of features rows, rescaling
    each feature to a specific range (often [0, 1])

    The rescaled value for a feature E is calculated as,

    Rescaled(ei) = (ei − Emin)∗(max − min)/(Emax − Emin) + min

    For the case Emax == Emin,  Rescaled(ei) = 0.5∗(max + min)

    """

    def transform(self, data, model, settings, nfrag):
        """MinMax Scaler Operation.

        :param data: A list with nfrag pandas's dataframe to perform
            the Normalization.
        :param settings: A dictionary that contains:
          - attributes: A list of features names to normalize;
          - min: Minimum value;
          - max: Maximum value;
          - alias: Aliases of the new columns (overwrite the fields if empty);
        :param nfrag: A number of fragments;
        :return: A list with nfrag pandas's dataframe
        """

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _minmax_scaler(data[f], settings, model)

        return result
    
    def fit(self, data, settings, nfrag):

        columns = settings.get('attributes', [])

        if len(columns) == 0:
            raise Exception('You must insert the fields to '
                            'perform the Normalization.')

        # generate a list of the min and the max element to each subset.
        minmax_partial = \
            [_agg_maxmin(data[f], columns) for f in range(nfrag)]

        # merge them into only one list
        minmax = mergeReduce(_merge_maxmin, minmax_partial)
        return minmax

    def transform_serial(self, data, model, settings):
        return _minmax_scaler_(data, settings, model)


@task(returns=list)
def _agg_maxmin(df, columns):
    """Generate a list of min and max values, excluding NaN values."""
    min_max_p = []
    for col in columns:
        p = [np.min(df[col].values, axis=0), np.max(df[col].values, axis=0)]
        min_max_p.append(p)
    return min_max_p


@task(returns=list)
def _merge_maxmin(minmax1, minmax2):
    """Merge min and max values."""
    minmax = []
    for feature in zip(minmax1, minmax2):
        di, dj = feature
        minimum = di[0] if di[0] < dj[0] else dj[0]
        maximum = di[1] if di[1] > dj[1] else dj[1]
        minmax.append([minimum, maximum])
    return minmax


@task(returns=list)
def _minmax_scaler(data, settings, minmax):
    return _minmax_scaler_(data, settings, minmax)


def _minmax_scaler_(data, settings, minmax):
    """Normalize by min max mode."""
    features = settings['attributes']
    alias = settings.get('alias', [])
    min_r = settings.get('min', 0)
    max_r = settings.get('max', 1)

    if len(alias) != len(features):
        alias = features

    for i, (alias, col) in enumerate(zip(alias, features)):
        minimum, maximum = minmax[i]
        data[alias] = data[col].apply(
                    lambda xs: calculation(xs, minimum, maximum, min_r, max_r))
    return data


def calculation(xs, minimum, maximum, min_r, max_r):
    features = []
    diff_r = float(max_r - min_r)
    for xi, mi, ma in zip(xs, minimum, maximum):
        if ma == mi:
            v = 0.5 * (max_r + min_r)
        else:
            v = (float(xi - mi) * (diff_r/(ma - mi))) + min_r
        features.append(v)
    return features


