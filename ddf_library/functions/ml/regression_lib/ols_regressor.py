#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF, generate_info
from ddf_library.ddf_model import ModelDDF

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object

import numpy as np
import pandas as pd


class OrdinaryLeastSquares(ModelDDF):
    """
    Linear regression is a linear model, e.g. a model that assumes a linear
    relationship between the input variables and the single output variable.
    More specifically, that y can be calculated from a linear combination of the
    input variables (x).

    When there is a single input variable (x), the method is referred to as
    simple linear regression. When there are multiple input variables,
    literature from statistics often refers to the method as multiple
    linear regression.

    b1 = (sum(x*y) + n*m_x*m_y) / (sum(x²) -n*(m_x²))
    b0 = m_y - b1*m_x

    :Example:

    >>> model = OrdinaryLeastSquares('features', 'y').fit(ddf1)
    >>> ddf2 = model.transform(ddf1)
    """

    def __init__(self, feature_col, label_col):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param mode: *'simple'* to use method of least squares (works only
         for 2-D data) or *'SDG'* to using Stochastic Gradient Descent;
        :param max_iter: Maximum number of iterations (default, 100);
        :param alpha: *'SDG'* learning rate parameter  (default, 0.01).
        """
        super(OrdinaryLeastSquares, self).__init__()

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col

        self.model = []
        self.name = 'OrdinaryLeastSquares'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        features = self.settings['feature_col']
        label = self.settings['label_col']

        cols = [features, label]

        partial_calculation = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            partial_calculation[f] = _lr_computation_xs(df[f], cols)

        calculation = merge_reduce(_lr_merge_info, partial_calculation)
        parameters = _lr_compute_line_2d(calculation)
        parameters = compss_wait_on(parameters)

        self.model = [parameters]
        return self

    def fit_transform(self, data, pred_col='pred_LinearReg'):
        """
        Fit the model and transform.

        :param data: DDF
        :param pred_col: Output prediction column (default, *'pred_LinearReg'*);
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, pred_col)

        return ddf

    def transform(self, data, pred_col='pred_LinearReg'):
        """
        :param data: DDF
        :param pred_col: Output prediction column (default, *'pred_LinearReg'*);
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        features = self.settings['feature_col']
        self.settings['pred_col'] = pred_col

        df, nfrag, tmp = self._ddf_initial_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _predict(df[f], features, pred_col,
                                          self.model[0], f)

        uuid_key = self._ddf_add_task(task_name='transform_ols',
                                      status='COMPLETED', lazy=self.OPT_OTHER,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)
        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=1)
def _lr_computation_xs(data, cols):
    """Partial calculation."""
    feature, label = cols
    data = data[cols].dropna()

    x = data[feature].values
    y = data[label].values
    del data

    sum_x = x.sum()
    square_x = (x**2).sum()
    col1 = [sum_x, len(x), square_x]

    sum_y = y.sum()
    square_y = (y**2).sum()
    col2 = [sum_y, len(y), square_y]

    xy = np.inner(x, y)
    return [col1, col2, [0, 0, xy]]


@task(returns=1)
def _lr_merge_info(info1, info2):
    """Merge calculation."""
    info = []
    for p1, p2 in zip(info1, info2):
        sum_t = p1[0] + p2[0]
        size_t = p1[1] + p2[1]
        square_t = p1[2] + p2[2]
        info.append([sum_t, size_t, square_t])
    return info


@task(returns=1)
def _lr_compute_line_2d(info):
    """Generate the regression."""
    rx, ry, rxy = info

    sum_x, n, square_x = rx
    sum_y, _, square_y = ry
    _, _, sum_xy = rxy

    m_x = sum_x/n
    m_y = sum_y/n
    b1 = (sum_xy - n * m_x * m_y)/(square_x - n * (m_x**2))

    b0 = m_y - b1 * m_x

    return [b0, b1]


@task(returns=2)
def _predict(data, x, y, model, frag):
    return _predict_(data, x, y, model, frag)


def _predict_(data, features, target, model, frag):
    """Predict the values."""

    n_rows = len(data)

    if n_rows > 0:
        xs = np.c_[np.ones(n_rows), data[features].values]
        tmp = np.dot(xs, model)
    else:
        tmp = np.nan

    if target in data.columns:
        data.drop([target], axis=1, inplace=True)

    data[target] = tmp

    info = generate_info(data, frag)
    return data, info
