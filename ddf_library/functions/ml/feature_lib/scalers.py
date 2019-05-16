#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF
from ddf_library.ddf_model import ModelDDF
from ddf_library.utils import generate_info

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object

import numpy as np
import pandas as pd


class MinMaxScaler(ModelDDF):
    """
    MinMaxScaler transforms a data set of features rows, rescaling
    each feature to a specific range (often [0, 1])

    :Example:

    >>> scaler = MinMaxScaler(input_col='features',
    >>>                       output_col='output').fit(ddf1)
    >>> ddf2 = scaler.transform(ddf1)
    """

    def __init__(self, input_col, feature_range=(0, 1), remove=False):
        """
        :param input_col: Column with the features;
        :param feature_range: A tuple with the range, default is (0,1);
        :param remove: Remove input columns after execution (default, False).
        """
        super(MinMaxScaler, self).__init__()

        if not isinstance(input_col, list):
            input_col = [input_col]

        if not isinstance(feature_range, tuple) or \
                feature_range[0] >= feature_range[1]:
            raise Exception("You must inform a valid `feature_range`.")

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['feature_range'] = feature_range
        self.settings['remove'] = remove

        self.model = []
        self.name = 'MinMaxScaler'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        columns = self.settings['input_col']
        # generate a list of the min and the max element to each subset.
        minmax_partial = \
            [_agg_maxmin(df[f], columns) for f in range(nfrag)]

        # merge them into only one list
        minmax = merge_reduce(_merge_maxmin, minmax_partial)
        minmax = compss_wait_on(minmax)
        compss_delete_object(minmax_partial)

        self.model = [minmax]
        return self

    def fit_transform(self, data, output_col=None):
        """
        Fit the model and transform.

        :param data: DDF
        :param output_col: Output column;
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, output_col)

        return ddf

    def transform(self, data, output_col=None):
        """
        :param data: DDF
        :param output_col: Output column;
        :return: DDF
        """

        input_col = self.settings['input_col']
        if not output_col:
            output_col = input_col

        if not isinstance(output_col, list):
            output_col = ['{}{}'.format(col, output_col) for col in input_col]

        self.settings['output_col'] = output_col

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        df, nfrag, tmp = self._ddf_initial_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _minmax_scaler(df[f], self.settings,
                                                self.model[0], f)

        uuid_key = self._ddf_add_task(task_name='task_transform_minmax_scaler',
                                      status='COMPLETED', opt=self.OPT_OTHER,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=1)
def _agg_maxmin(df, columns):
    """Generate a list of min and max values, excluding NaN values."""
    min_max_p = []
    if len(df) > 0:
        min_max_p = [np.min(df[columns].values, axis=0),
                     np.max(df[columns].values, axis=0)]

    return min_max_p


@task(returns=1)
def _merge_maxmin(info1, info2):
    """Merge min and max values."""

    if len(info1) > 0 and len(info2) > 0:
        minimum = np.min([info1[0], info2[0]], axis=0)
        maximum = np.max([info1[1], info2[1]], axis=0)
        info = [minimum, maximum]
    elif len(info1) == 0:
        info = info2
    else:
        info = info1
    return info


@task(returns=2)
def _minmax_scaler(data, settings, info, frag):
    """Normalize by min max mode."""
    features = settings['input_col']
    alias = settings.get('output_col', [])
    min_r, max_r = settings.get('feature_range', (0, 1))
    remove_input = settings.get('remove', False)

    if len(alias) != len(features):
        alias = features

    values = data[features].values
    to_remove = [c for c in alias if c in data.columns]
    if remove_input:
        to_remove += input_col
    data.drop(to_remove, axis=1, inplace=True)

    if len(data) > 0:

        from sklearn.preprocessing import MinMaxScaler

        minimum, maximum = info
        minimum = np.array(minimum)
        maximum = np.array(maximum)

        scale_ = (max_r - min_r) / (maximum - minimum)

        scaler = MinMaxScaler()
        scaler.data_min_ = minimum
        scaler.data_max_ = maximum
        scaler.scale_ = scale_
        scaler.data_range_ = maximum - minimum
        scaler.min_ = min_r - minimum * scale_

        res = scaler.transform(values)
        del values

        data = pd.concat([data, pd.DataFrame(res, columns=alias)], axis=1)

    else:
        for col in alias:
            data[col] = np.nan

    info = generate_info(data, frag)
    return data, info


class MaxAbsScaler(ModelDDF):
    """
    MaxAbsScaler transforms a data set of features rows,
    rescaling each feature to range [-1, 1] by dividing through
    the maximum absolute value in each feature.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the
    training set will be 1.0. It does not shift/center the data,
    and thus does not destroy any sparsity.

    :Example:

    >>> scaler = MaxAbsScaler(input_col='features',
    >>>                       output_col='features_norm').fit(ddf1)
    >>> ddf2 = scaler.transform(ddf1)
    """

    def __init__(self, input_col, remove=False):
        """
        :param input_col: Column with the features;
        :param output_col: Output column;
        :param remove: Remove input columns after execution (default, False).
        """
        super(MaxAbsScaler, self).__init__()

        if not isinstance(input_col, list):
            input_col = [input_col]

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['remove'] = remove

        self.model = {}
        self.name = 'MaxAbsScaler'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        columns = self.settings['input_col']
        # generate a list of the min and the max element to each subset.
        minmax_partial = \
            [_agg_maxmin(df[f], columns) for f in range(nfrag)]

        # merge them into only one list
        minmax = merge_reduce(_merge_maxmin, minmax_partial)
        minmax = compss_wait_on(minmax)
        compss_delete_object(minmax_partial)

        self.model = [minmax]
        return self

    def fit_transform(self, data, output_col=None):
        """
        Fit the model and transform.

        :param data: DDF
        :param output_col: Output column;
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, output_col)

        return ddf

    def transform(self, data, output_col=None):
        """
        :param data: DDF
        :param output_col: Output column;
        :return: DDF
        """

        input_col = self.settings['input_col']
        if not output_col:
            output_col = input_col

        if not isinstance(output_col, list):
            output_col = ['{}{}'.format(col, output_col) for col in input_col]

        self.settings['output_col'] = output_col

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        df, nfrag, tmp = self._ddf_initial_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]

        for f in range(nfrag):
            result[f], info[f] = _maxabs_scaler(df[f], self.model[0],
                                                self.settings, f)

        uuid_key = self._ddf_add_task(task_name='task_transform_maxabs_scaler',
                                      status='COMPLETED', opt=self.OPT_OTHER,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=2)
def _maxabs_scaler(data, minmax, settings, frag):
    """Normalize by range mode."""
    features = settings['input_col']
    alias = settings.get('output_col', [])
    remove_input = settings.get('remove', False)

    if len(alias) != len(features):
        alias = features

    values = data[features].values
    to_remove = [c for c in alias if c in data.columns]
    if remove_input:
        to_remove += input_col
    data.drop(to_remove, axis=1, inplace=True)

    if len(data) > 0:

        from sklearn.preprocessing import MaxAbsScaler

        minimum, maximum = minmax
        minimum = np.abs(minimum)
        maximum = np.abs(maximum)

        max_abs = np.max([minimum, maximum], axis=0)
        scale = max_abs.copy()
        scale[scale == 0.0] = 1.0

        scaler = MaxAbsScaler()
        scaler.n_samples_seen_ = len(values)
        scaler.max_abs_ = max_abs
        scaler.scale_ = scale

        res = scaler.transform(values)
        del values

        data = pd.concat([data, pd.DataFrame(res, columns=alias)], axis=1)

    else:
        for col in alias:
            data[col] = np.nan

    info = generate_info(data, frag)
    return data, info


class StandardScaler(ModelDDF):
    """
    The standard score of a sample x is calculated as:

        z = (x - u) / s

    where u is the mean of the training samples or zero if
    with_mean=False, and s is the standard deviation of the
    training samples or one if with_std=False.

    :Example:

    >>> scaler = StandardScaler(input_col='features',
    >>>                         output_col='norm').fit(ddf1)
    >>> ddf2 = scaler.transform(ddf1)
    """

    def __init__(self, input_col, with_mean=True, with_std=True, remove=False):
        """
        :param input_col: Column with the features;
        :param output_col: Output column;
        :param with_mean: True to use the mean (default is True);
        :param with_std: True to use standard deviation of the
         training samples (default is True);
        :param remove: Remove input columns after execution (default, False).
        """
        super(StandardScaler, self).__init__()

        if not isinstance(input_col, list):
            input_col = [input_col]

        self.settings = dict()
        self.settings['input_col'] = input_col
        self.settings['with_mean'] = with_mean
        self.settings['with_std'] = with_std
        self.settings['remove'] = remove

        self.model = []
        self.name = 'StandardScaler'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        features = self.settings['input_col']

        # compute the sum of each subset column
        data_set = [[] for _ in range(nfrag)]
        sums = [[] for _ in range(nfrag)]
        sse = [[] for _ in range(nfrag)]

        for f in range(nfrag):
            data_set[f], sums[f] = _agg_sum(df[f], features)
        # merge then to compute a mean
        mean = merge_reduce(_merge_sum, sums)

        # using this mean, compute the variance of each subset column
        for f in range(nfrag):
            # noinspection PyTypeChecker
            sse[f] = _agg_sse(data_set[f], mean)
        merged_sse = merge_reduce(_merge_sse, sse)

        mean = compss_wait_on(mean)
        sse = compss_wait_on(merged_sse)

        compss_delete_object(data_set)
        compss_delete_object(sums)
        compss_delete_object(sse)
        self.model = [[mean, sse]]

        return self

    def fit_transform(self, data, output_col=None):
        """
        Fit the model and transform.

        :param data: DDF
        :param output_col: Output column;
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, output_col)

        return ddf

    def transform(self, data, output_col=None):
        """
        :param data: DDF
        :param output_col: Output column;
        :return: DDF
        """

        input_col = self.settings['input_col']
        if not output_col:
            output_col = input_col

        if not isinstance(output_col, list):
            output_col = ['{}{}'.format(col, output_col) for col in input_col]

        self.settings['output_col'] = output_col

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        df, nfrag, tmp = self._ddf_initial_setup(data)

        mean, sse = self.model[0]
        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _standard_scaler(df[f], self.settings,
                                                  mean, sse, f)

        uuid_key = self._ddf_add_task(task_name='transform_standard_scaler',
                                      status='COMPLETED', opt=self.OPT_OTHER,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=2)
def _agg_sum(df, features):
    """Pre-compute some values."""

    df = df[features].values
    sum_partial = [np.nansum(df, axis=0), len(df)]

    return df, sum_partial


@task(returns=1, priority=True)
def _merge_sum(sum1, sum2):
    """Merge pre-computation."""
    count = sum1[1] + sum2[1]
    sums = np.add(sum1[0], sum2[0])
    sum_count = [sums, count]

    return sum_count


@task(returns=1)
def _agg_sse(df, sum_count):
    """Perform a partial SSE calculation."""

    means = np.array(sum_count[0]) / sum_count[1]
    sum_sse = np.sum((df - means)**2, axis=0)

    return sum_sse


@task(returns=1, priority=True)
def _merge_sse(sum1, sum2):
    """Merge the partial SSE."""
    sum_count = sum1 + sum2
    return sum_count


@task(returns=2)
def _standard_scaler(data, settings, mean, sse, frag):
    """Normalize by Standard mode."""
    features = settings['input_col']
    alias = settings['output_col']
    with_mean = settings['with_mean']
    with_std = settings['with_std']
    remove_input = settings.get('remove', False)

    if len(alias) != len(features):
        alias = features

    values = data[features].values
    to_remove = [c for c in alias if c in data.columns]
    if remove_input:
        to_remove += input_col
    data.drop(to_remove, axis=1, inplace=True)

    if len(data) > 0:

        from sklearn.preprocessing import StandardScaler

        size = mean[1]
        var_ = np.array(sse) / size
        mean_ = np.array(mean[0]) / size

        scaler = StandardScaler()
        scaler.mean_ = mean_ if with_mean else None
        scaler.scale_ = np.sqrt(var_) if with_std else None
        scaler.var_ = var_ if with_std else None
        scaler.n_samples_seen_ = size

        res = scaler.transform(values)
        del values

        data = pd.concat([data, pd.DataFrame(res, columns=alias)], axis=1)

    else:
        for col in alias:
            data[col] = np.nan

    info = generate_info(data, frag)
    return data, info
