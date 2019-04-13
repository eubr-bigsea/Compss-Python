#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF, generate_info
from ddf_library.ddf_model import ModelDDF

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object

import math
import numpy as np
import pandas as pd

__all__ = ['OrdinaryLeastSquares', 'SGDRegressor']


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

    def __init__(self, feature_col, label_col, pred_col='pred_LinearReg'):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param pred_col: Output prediction column (default, *'pred_LinearReg'*);
        :param mode: *'simple'* to use method of least squares (works only
         for 2-D data) or *'SDG'* to using Stochastic Gradient Descent;
        :param max_iter: Maximum number of iterations (default, 100);
        :param alpha: *'SDG'* learning rate parameter  (default, 0.01).
        """
        super(OrdinaryLeastSquares, self).__init__()

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['pred_col'] = pred_col

        self.model = []
        self.name = 'OrdinaryLeastSquares'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        features = self.settings['feature_col']
        label = self.settings['label_col']

        cols = [features, label]

        calcs = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            calcs[f] = _lr_computation_xs(df[f], cols)

        calcs = merge_reduce(_lr_merge_calcs, calcs)
        parameters = _lr_compute_line_2d(calcs)
        parameters = compss_wait_on(parameters)

        self.model = [parameters]
        return self

    def fit_transform(self, data):
        """
        Fit the model and transform.

        :param data: DDF
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data)

        return ddf

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        features = self.settings['feature_col']
        pred_col = self.settings['pred_col']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _predict(df[f], features, pred_col,
                                          self.model[0], f)

        uuid_key = self._ddf_add_task(task_name='transform_ols',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)
        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


class SGDRegressor(ModelDDF):
    """
    Linear model fitted by minimizing a regularized empirical loss with SGD.
    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way
    with a decreasing strength schedule (aka learning rate).

    :Example:

    >>> model = SGDRegressor('features', 'y').fit(ddf1)
    >>> ddf2 = model.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, pred_col='pred_LinearReg',
                 max_iter=100, alpha=1):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param pred_col: Output prediction column (default, *'pred_LinearReg'*);
        :param max_iter: Maximum number of iterations (default, 100);
        :param alpha: learning rate parameter  (default, 1). This method sets
         the learning rate parameter used by Gradient Descent when updating
         the hypothesis after each iteration. Up to a point, higher values will
         cause the algorithm to converge on the optimal solution more quickly,
         however if the value is set too high then it will fail to converge at
         all, yielding successively larger errors on each iteration.
        """
        super(SGDRegressor, self).__init__()

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['pred_col'] = pred_col
        self.settings['max_iter'] = max_iter
        self.settings['alpha'] = alpha

        self.model = []
        self.name = 'SGDRegressor'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        features = self.settings['feature_col']
        label = self.settings['label_col']
        alpha = self.settings['alpha']
        max_iter = self.settings['max_iter']

        parameters = _gradient_descent(df, features, label,
                                       alpha, max_iter, nfrag)

        parameters = compss_wait_on(parameters)
        self.model = [parameters]
        return self

    def fit_transform(self, data):
        """
        Fit the model and transform.

        :param data: DDF
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data)

        return ddf

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        features = self.settings['feature_col']
        pred_col = self.settings['pred_col']

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _predict(df[f], features, pred_col,
                                          self.model[0], f)

        uuid_key = self._ddf_add_task(task_name='sgd_regressor',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)
        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


# --------------
# Simple Linear Regression


@task(returns=1)
def _lr_computation_xs(data, cols):
    """Partial calculation."""
    col1, col2 = cols
    x = np.array(data[col1].tolist()).flatten()
    y = data[col2].values
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
def _lr_merge_calcs(calcs1, calcs2):
    """Merge calculation."""
    calcs = []
    for p1, p2 in zip(calcs1, calcs2):
        sum_t = p1[0] + p2[0]
        size_t = p1[1] + p2[1]
        square_t = p1[2] + p2[2]
        calcs.append([sum_t, size_t, square_t])
    return calcs


@task(returns=1)
def _lr_compute_line_2d(calcs):
    """Generate the regression."""
    rx, ry, rxy = calcs

    sum_x, n, square_x = rx
    sum_y, _, square_y = ry
    _, _, sum_xy = rxy

    m_x = sum_x/n
    m_y = sum_y/n
    b1 = (sum_xy - n * m_x * m_y)/(square_x - n * (m_x**2))

    b0 = m_y - b1 * m_x

    return [b0, b1]

# --------------
# SGD regressor:


def _gradient_descent(data, features, label, alpha, iters, nfrag):
    """Regression by using gradient Descent."""
    theta = np.random.randn(3)

    input_data = [select_att(data[f], [features, label])
                  for f in range(nfrag)]

    for i in range(iters):
        stage1 = [_lr_sgb_first_stage(input_data[f], theta)
                  for f in range(nfrag)]
        grad = merge_reduce(_lr_sgb_agg, stage1)
        compss_delete_object(stage1)
        theta = _lr_sgb_theta_computation(grad, alpha)

    return theta


@task(returns=1)
def select_att(data, cols):
    features, label = cols
    features = np.array(data[features].tolist())
    labels = data[label].values
    del data
    return [features, labels]


@task(returns=1)
def _lr_sgb_first_stage(data, theta):
    """Peform the partial gradient generation."""

    features, label = data

    size = len(label)

    if size > 0:

        shapes = features.shape
        if len(shapes) == 2:
            dim = shapes[1]
        else:
            dim = 1

        if (dim+1) != len(theta):
            theta = np.random.randn(dim+1)

        gradient = [0 for _ in range(dim+1)]

        # Translates slice objects to concatenation along the second axis.
        xs = np.c_[np.ones(size), features]
        partial_error = np.dot(xs, theta.T) - label

        for j in range(dim+1):
            grad = np.multiply(partial_error, xs[:, j])
            gradient[j] += np.sum(grad)

        return [gradient, size, dim, theta]

    return [0, 0, -1, 0]


@task(returns=1)
def _lr_sgb_agg(error1, error2):
    """Merge the partial gradients."""
    grad1, size1, dim1, theta1 = error1
    grad2, size2, dim2, theta2 = error2

    theta = theta1
    dim = dim1
    if dim2 > dim1:
        # meaning that partition1 is empty
        dim = dim2
        theta = theta2

    sum_grad = np.sum([grad1, grad2], axis=0)
    size = size1 + size2

    return [sum_grad, size, dim, theta]


# @local
def _lr_sgb_theta_computation(info, alpha):
    """Generate new theta."""
    info = compss_wait_on(info)
    grad, size, dim, theta = info

    for j in range(dim+1):
        theta[j] = theta[j] - ((alpha / size) * grad[j])

    return theta


@task(returns=2)
def _predict(data, x, y, model, frag):
    return _predict_(data, x, y, model, frag)


def _predict_(data, features, target, model, frag):
    """Predict the values."""
    tmp = []
    if len(data) > 0:
        if isinstance(data.iloc[0][features], list):
            dim = len(data.iloc[0][features])
        else:
            dim = 1

        if dim > 1:
            n_rows = len(data)
            model = np.array(model)
            features = np.array(data[features].tolist())
            xs = np.c_[np.ones(n_rows), features]
            del features
            tmp = np.dot(xs, model)

            # for row in data[features].values:
            #     y = model[0]
            #     for j in range(len(row)-1):
            #         y += row[j] * model[j+1]
            #     tmp.append(y)
            #
        else:
            xys = np.array(data[features].tolist()).flatten()
            tmp = [model[0] + model[1]*row for row in xys]

    data[target] = tmp

    info = generate_info(data, frag)
    return data, info
