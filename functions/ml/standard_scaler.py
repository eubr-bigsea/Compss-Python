#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np



class StandardScalerOperation(object):
    """Standard Scaler Operation.

    Scales the data to unit standard deviation.
    """

    def transform(self, data, model, settings, nfrag):
        """Standard Scaler Operation.

        :param data: A list with nfrag pandas's dataframe to perform
            the Normalization.
        :param settings: A dictionary that contains:
          - attributes: A list of columns names to normalize;
          - alias: Aliases of the new columns (overwrite the fields if empty);
        :param nfrag: A number of fragments;
        :return: A list with nfrag pandas's dataframe
        """

        mean, sse = model
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _stardard_scaler(data[f], settings, mean, sse)

        return result
    
    def fit(self, data, settings, nfrag):
        features = settings.get('attributes', [])
        settings['alias'] = settings.get('alias', [])
        settings['attributes'] = features

        if len(features) == 0:
            raise Exception('You must insert the fields to '
                            'perform the Normalization.')

        # compute the sum of each subset column
        sum_partial = \
            [_agg_sum(data[f], features) for f in range(nfrag)]
        # merge then to compute a mean
        mean = mergeReduce(_merge_sum, sum_partial)
        # using this mean, compute the variance of each subset column
        sse_partial = \
            [_agg_sse(data[f], features, mean) for f in range(nfrag)]
        sse = mergeReduce(_merge_sse, sse_partial)
        return [mean, sse]

    def transform_serial(self, data, settings, model):
        mean, sse = model
        return _stardard_scaler_(data, settings, mean, sse)


@task(returns=list)
def _agg_sum(df, features):
    """Pre-compute some values."""
    sum_partial = []
    for feature in features:
        sum_p = [np.nansum(df[feature].values.tolist(), axis=0),
                 len(df[feature])]
        sum_partial.append(sum_p)
    return sum_partial


@task(returns=list, priority=True)
def _merge_sum(sum1, sum2):
    """Merge pre-computation."""
    sum_count = []
    for f_i, f_j in zip(sum1, sum2):
        count = f_i[1] + f_j[1]
        sums = []
        for di, dj in zip(f_i[0], f_j[0]):
            sum_f = di + dj
            sums.append(sum_f)
        sum_count.append([sums, count])

    return sum_count


@task(returns=list)
def _agg_sse(df, features, sum_count):
    """Perform a partial SSE calculation."""
    sum_sse = []

    for sum_f, col in zip(sum_count, features):
        size = sum_f[1]
        sums = sum_f[0]
        means = [x / size for x in sums]
        sum_sse.append(
                np.nansum(df[col].apply(lambda xs: computation_sse(xs, means))
                          .values.tolist(), axis=0))

    return sum_sse


def computation_sse(xs, means):
    sse = []
    for xi, mi in zip(xs, means):
        sse.append((xi-mi)**2)
    return sse


@task(returns=list, priority=True)
def _merge_sse(sum1, sum2):
    """Merge the partial SSE."""
    sum_count = []
    for di, dj in zip(sum1, sum2):
        sum_count.append(di+dj)
    return sum_count


@task(returns=list)
def _stardard_scaler(data, settings, mean, sse):
    return _stardard_scaler_(data, settings, mean, sse)


def _stardard_scaler_(data, settings, mean, sse):
    """Normalize by Standard mode."""
    features = settings['attributes']
    alias = settings['alias']
    if len(alias) != len(features):
        alias = features

    for i, (alias, col) in enumerate(zip(alias, features)):
        print mean[i]
        size = mean[i][1]
        means = [x / size for x in mean[i][0]]
        stds = [np.sqrt(sse_p/size) for sse_p in sse[i]]  # std population
        print stds
        data[alias] = data[col]\
            .apply(lambda xs: computation_scaler(xs, means, stds))

    return data


def computation_scaler(xs, means, stds):
    scaler = []
    for xi, mi, std in zip(xs, means, stds):
        scaler.append(float(xi - mi)/std)
    return scaler
