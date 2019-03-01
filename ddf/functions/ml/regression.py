#!/usr/bin/python
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import math
import numpy as np
import pandas as pd
from pycompss.api.task import task
from pycompss.api.local import *
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
from ddf.ddf import DDF
from ddf.ddf_model import ModelDDF

__all__ = ['LinearRegression']

import sys
sys.path.append('../../')


class LinearRegression(ModelDDF):
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

    >>> model = LinearRegression('features', 'y').fit(ddf1)
    >>> ddf2 = model.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, pred_col='pred_LinearReg',
                 mode='SDG', max_iter=100, alpha=0.01):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param pred_col: Output prediction column (default, *'pred_LinearReg'*);
        :param mode: *'simple'* to use method of least squares (works only
         for 2-D data) or *'SDG'* to using Stochastic Gradient Descent;
        :param max_iter: Maximum number of iterations (default, 100);
        :param alpha: *'SDG'* learning rate parameter  (default, 0.01).
        """
        super(LinearRegression, self).__init__()

        if mode not in ['simple', 'SDG']:
            raise Exception("You must inform a valid `mode`.")

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['pred_col'] = pred_col
        self.settings['mode'] = mode
        self.settings['max_iter'] = max_iter
        self.settings['alpha'] = alpha

        self.model = []
        self.name = 'LinearRegression'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        features = self.settings['feature_col']
        label = self.settings['label_col']
        mode = self.settings.get('mode', 'SDG')

        if mode not in ['simple', 'SDG']:
            raise Exception("You must inform a valid `mode`.")

        parameters = 0

        if mode == "SDG":
            alpha = self.settings['alpha']
            max_iter = self.settings['max_iter']

            parameters = _gradient_descent(df, features, label,
                                           alpha, max_iter, nfrag)
        elif mode == 'simple':
            """
                Simple Linear Regression: This mode is useful only if
                you have a small dataset.
            """
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
                                          self.model[0])

        uuid_key = self._ddf_add_task(task_name='task_transform_linear_reg',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)  #TODO: dont need !?
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


# --------------
# Simple Linear Regression


@task(returns=1)
def _lr_computation_xs(data, cols):
    """Partial calculation."""
    col1, col2 = cols
    x = np.array(data[col1].tolist()).flatten()

    sum_x = x.sum()
    square_x = (x**2).sum()
    col1 = [sum_x, len(x), square_x]

    y = data[col2].values
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

    m_x = float(sum_x)/n
    m_y = float(sum_y)/n
    b1 = float(sum_xy - n*m_x*m_y)/(square_x - n * (m_x**2))

    b0 = m_y - b1*m_x

    return [b0, b1]

# --------------
# SGD mode:


def _gradient_descent(data, features, label, alpha, iters, nfrag):
    """Regression by using gradient Descent."""
    theta = np.array([0, 0, 0])

    for i in range(iters):
        stage1 = [_lr_sgb_first_stage(data[f], [features, label], theta)
                  for f in range(nfrag)]
        grad = merge_reduce(_lr_sgb_agg, stage1)
        theta = _lr_sgb_theta_computation(grad, alpha)

    return theta


@task(returns=1)
def _lr_sgb_first_stage(data, cols, theta):
    """Peform the partial gradient generation."""
    size = len(data)
    features, label = cols

    if size > 0:
        if isinstance(data.iloc[0][features], list):
            dim = len(data.iloc[0][features])
        else:
            dim = 1

        if (dim+1) != len(theta):
            theta = np.array([0 for _ in range(dim+1)])

        # Translates slice objects to concatenation along the second axis.
        xs = np.c_[np.ones(size), np.array(data[features].tolist())]
        partial_error = np.dot(xs, theta.T) - data[label].values

        for j in range(dim+1):
            grad = np.multiply(partial_error, xs[:, j])

        return [np.sum(grad), size, dim, theta]

    return [0, 0, -1, 0]


@task(returns=list)
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

    sum_grad = grad1 + grad2
    size = size1 + size2

    return [sum_grad, size, dim, theta]


@local
def _lr_sgb_theta_computation(info, alpha):
    """Generate new theta."""
    grad, size, dim, theta = info

    temp = np.zeros(theta.shape)
    for j in range(dim+1):
        temp[j] = theta[j] - ((float(alpha) / size) * grad)

    return temp


@task(returns=2)
def _predict(data, x, y, model):
    return _predict_(data, x, y, model)


def _predict_(data, features, target, model):
    """Predict the values."""
    tmp = []
    if len(data) > 0:
        if isinstance(data.iloc[0][features], list):
            dim = len(data.iloc[0][features])
        else:
            dim = 1

        if dim > 1:
            for row in data[features].values:
                y = model[0]
                for j in range(len(row)-1):
                    y += row[j]*model[j+1]
                tmp.append(y)
        else:
            xys = np.array(data[features].tolist()).flatten()
            tmp = [model[0] + model[1]*row for row in xys]

    data[target] = tmp

    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
    return data, info
