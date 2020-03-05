#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.bases.context_base import ContextBase
from ddf_library.bases.metadata import OPTGroup
from ddf_library.ddf import DDF
from ddf_library.bases.ddf_model import ModelDDF
from ddf_library.utils import generate_info,  read_stage_file

from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object

import numpy as np
import pandas as pd


class MaxAbsScaler(ModelDDF):
    # noinspection PyUnresolvedReferences
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

    def __init__(self):
        """
        """
        super(MaxAbsScaler, self).__init__()

    def fit(self, data, input_col):
        """
        Fit the model.

        :param data: DDF
        :param input_col: Column with the features;
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        if not isinstance(input_col, list):
            input_col = [input_col]
        self.input_col = input_col

        # generate a list of the min and the max element to each subset.
        minmax_partial = \
            [_agg_maxmin(df[f], self.input_col) for f in range(nfrag)]

        # merge them into only one list
        minmax = merge_reduce(_merge_maxmin, minmax_partial)
        minmax = compss_wait_on(minmax)
        compss_delete_object(minmax_partial)

        self.model = {'model': minmax, 'algorithm': self.name}
        return self

    def fit_transform(self, data, input_col, output_col=None, remove=False):
        """
        Fit the model and transform.

        :param data: DDF
        :param input_col: Column with the features;
        :param output_col: Output column;
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, output_col=output_col, remove=remove)

        return ddf

    def transform(self, data, input_col=None, output_col=None, remove=False):
        """
        :param data: DDF
        :param input_col: Column with the features;
        :param output_col: Output column;
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.check_fitted_model()
        if input_col:
            self.input_col = input_col
        self.remove = remove

        if not output_col:
            output_col = self.input_col

        elif not isinstance(output_col, list):
            # noinspection PyTypeChecker
            output_col = ['{}{}'.format(col, output_col)
                          for col in self.input_col]

        self.output_col = output_col

        settings = self.__dict__.copy()
        settings['model'] = settings['model']['model']

        def task_maxabs_scaler(df, params):
            return _maxabs_scaler(df, params)

        uuid_key = ContextBase\
            .ddf_add_task(self.name, opt=OPTGroup.OPT_SERIAL,
                          function=[task_maxabs_scaler, settings],
                          parent=[data.last_uuid])

        return DDF(last_uuid=uuid_key)


def _maxabs_scaler(data, settings):
    """Normalize by range mode."""
    features = settings['input_col']
    alias = settings.get('output_col', [])
    remove_input = settings.get('remove', False)
    minmax = settings['model']
    frag = settings['id_frag']

    if len(alias) != len(features):
        alias = features

    values = data[features].values
    to_remove = [c for c in alias if c in data.columns]
    if remove_input:
        to_remove += features
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


class MinMaxScaler(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    MinMaxScaler transforms a data set of features rows, rescaling
    each feature to a specific range (often [0, 1])

    :Example:

    >>> scaler = MinMaxScaler()
    >>> ddf2 = scaler.fit_transform(ddf1, input_col=['col1', 'col2'])
    """

    def __init__(self, feature_range=(0, 1)):
        """
        :param feature_range: A tuple with the range, default is (0,1);
        """
        super(MinMaxScaler, self).__init__()

        if not isinstance(feature_range, tuple) or \
                feature_range[0] >= feature_range[1]:
            raise Exception("You must inform a valid `feature_range`.")

        self.feature_range = feature_range

    def fit(self, data, input_col):
        """
        Fit the model.

        :param data: DDF
        :param input_col: Column with the features;
        :return: trained model
        """

        if not isinstance(input_col, list):
            input_col = [input_col]

        self.input_col = input_col
        df, nfrag, tmp = self._ddf_initial_setup(data)

        # generate a list of the min and the max element to each subset.
        minmax_partial = \
            [_agg_maxmin(df[f], self.input_col) for f in range(nfrag)]

        # merge them into only one list
        minmax = merge_reduce(_merge_maxmin, minmax_partial)
        minmax = compss_wait_on(minmax)
        compss_delete_object(minmax_partial)

        self.model = {'model': minmax, 'algorithm': self.name}
        return self

    def fit_transform(self, data, input_col, output_col=None, remove=False):
        """
        Fit the model and transform.

        :param data: DDF
        :param input_col: Column with the features;
        :param output_col: Output column;
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, output_col=output_col, remove=remove)

        return ddf

    def transform(self, data, input_col=None, output_col=None, remove=False):
        """
        :param data: DDF
        :param input_col: Column with the features;
        :param output_col: Output column;
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.check_fitted_model()
        if input_col:
            self.input_col = input_col
        self.remove = remove

        if not output_col:
            output_col = self.input_col

        elif not isinstance(output_col, list):
            # noinspection PyTypeChecker
            output_col = ['{}{}'.format(col, output_col)
                          for col in self.input_col]

        self.output_col = output_col

        settings = self.__dict__.copy()
        settings['model'] = settings['model']['model']

        def task_minmax_scaler(df, params):
            return _minmax_scaler(df, params)

        uuid_key = ContextBase \
            .ddf_add_task(self.name, opt=OPTGroup.OPT_SERIAL,
                          function=[task_minmax_scaler, settings],
                          parent=[data.last_uuid])

        return DDF(task_list=data.task_list, last_uuid=uuid_key)


@task(returns=1, data_input=FILE_IN)
def _agg_maxmin(data_input, columns):
    """Generate a list of min and max values, excluding NaN values."""
    df = read_stage_file(data_input, columns)
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


def _minmax_scaler(data, settings):
    """Normalize by min max mode."""
    info = settings['model']
    frag = settings['id_frag']
    features = settings['input_col']
    alias = settings.get('output_col', [])
    min_r, max_r = settings.get('feature_range', (0, 1))
    remove_input = settings.get('remove', False)

    if len(alias) != len(features):
        alias = features

    values = data[features].values
    to_remove = [c for c in alias if c in data.columns]
    if remove_input:
        to_remove += features
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


class StandardScaler(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    The standard score of a sample x is calculated as:

        z = (x - u) / s

    where u is the mean of the training samples or zero if
    with_mean=False, and s is the standard deviation of the
    training samples or one if with_std=False.

    :Example:

    >>> scaler = StandardScaler(with_mean=True, with_std=True)
    >>> ddf2 = scaler.fit_transform(ddf1, input_col=['col1', 'col2'])
    """

    def __init__(self, with_mean=True, with_std=True):
        """

        :param with_mean: True to use the mean (default is True);
        :param with_std: True to use standard deviation of the
         training samples (default is True);
        """
        super(StandardScaler, self).__init__()
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, data, input_col):
        """
        Fit the model.

        :param data: DDF
        :param input_col: Column with the features;
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        if not isinstance(input_col, list):
            input_col = [input_col]
        # noinspection PyTypeChecker
        self.input_col = input_col

        # compute the sum of each subset column
        sums = [0] * nfrag
        sse = [0] * nfrag

        for f in range(nfrag):
            sums[f] = _agg_sum(df[f], self.input_col)
        # merge then to compute a mean
        mean = merge_reduce(_merge_sum, sums)

        # using this mean, compute the variance of each subset column
        for f in range(nfrag):
            # noinspection PyTypeChecker
            sse[f] = _agg_sse(df[f], mean, self.input_col)
        merged_sse = merge_reduce(_merge_sse, sse)

        mean = compss_wait_on(mean)
        sse = compss_wait_on(merged_sse)

        compss_delete_object(sums)
        compss_delete_object(sse)
        self.model = {'model': [mean, sse], 'algorithm': self.name}

        return self

    def fit_transform(self, data, input_col, output_col=None, remove=False):
        """
        Fit the model and transform.

        :param data: DDF
        :param input_col: Column with the features;
        :param output_col: Output column;
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.fit(data, input_col)
        ddf = self.transform(data, output_col=output_col, remove=remove)

        return ddf

    def transform(self, data, input_col=None, output_col=None, remove=False):
        """
        :param data: DDF
        :param input_col: Column with the features;
        :param output_col: Output column;
        :param remove: Remove input columns after execution (default, False).
        :return: DDF
        """

        self.check_fitted_model()
        if input_col:
            self.input_col = input_col
        self.remove = remove

        if not output_col:
            output_col = self.input_col

        elif not isinstance(output_col, list):
            # noinspection PyTypeChecker
            output_col = ['{}{}'.format(col, output_col)
                          for col in self.input_col]

        self.output_col = output_col

        settings = self.__dict__.copy()
        settings['model'] = settings['model']['model']

        def task_standard_scaler(df, params):
            return _standard_scaler(df, params)

        uuid_key = ContextBase\
            .ddf_add_task(self.name, opt=OPTGroup.OPT_SERIAL,
                          function=[task_standard_scaler, settings],
                          parent=[data.last_uuid])

        return DDF(task_list=data.task_list, last_uuid=uuid_key)


@task(returns=1, data_input=FILE_IN)
def _agg_sum(data_input, features):
    """Pre-compute some values."""
    df = read_stage_file(data_input, features)
    sum_partial = [np.nansum(df[features].values, axis=0), len(df)]
    return sum_partial


@task(returns=1, priority=True)
def _merge_sum(sum1, sum2):
    """Merge pre-computation."""
    count = sum1[1] + sum2[1]
    sums = np.add(sum1[0], sum2[0])
    sum_count = [sums, count]

    return sum_count


@task(returns=1, data_input=FILE_IN)
def _agg_sse(data_input, sum_count, features):
    """Perform a partial SSE calculation."""
    df = read_stage_file(data_input, features)
    df = df[features].values
    means = np.array(sum_count[0]) / sum_count[1]
    sum_sse = np.sum((df - means)**2, axis=0)

    return sum_sse


@task(returns=1, priority=True)
def _merge_sse(sum1, sum2):
    """Merge the partial SSE."""
    sum_count = sum1 + sum2
    return sum_count


def _standard_scaler(data, settings):
    """Normalize by Standard mode."""

    frag = settings['id_frag']
    mean, sse = settings['model']
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
        to_remove += features
    data.drop(to_remove, axis=1, inplace=True)

    if len(data) > 0:

        from sklearn.preprocessing import StandardScaler

        size = mean[1]
        var_ = np.array(sse) / size
        mean_ = np.array(mean[0]) / size

        scaler = StandardScaler()
        scaler.mean_ = mean_ if with_mean else None
        if with_std:
            scaler.scale_ = np.sqrt(var_)
            scaler.mean_ = mean_
            scaler.var_ = var_ if with_std else None

        else:
            scaler.scale_ = None

        scaler.n_samples_seen_ = size
        res = scaler.transform(values)
        del values

        data = pd.concat([data, pd.DataFrame(res, columns=alias)], axis=1)

    else:
        for col in alias:
            data[col] = np.nan

    info = generate_info(data, frag)
    return data, info
