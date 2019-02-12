#!/usr/bin/python
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import math
import numpy as np
import pandas as pd
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
from ddf.ddf import COMPSsContext, DDF, ModelDDS

__all__ = ['LinearRegression']


import uuid
import sys
sys.path.append('../../')


class LinearRegression(object):
    """Linear regression.

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
    """

    def __init__(self, feature_col, label_col, pred_col='pred_LinearReg',
                 mode='SDG', max_iter=100, alpha=0.01):
        if not feature_col:
            raise Exception("You must inform the `features` field.")

        if not label_col:
            raise Exception("You must inform the `label` field.")

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

        :param data: DDF
        :return: trained model
        """

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)
        features = self.settings['feature_col']
        label = self.settings['label_col']
        mode = self.settings.get('mode', 'SDG')

        if mode not in ['simple', 'SDG']:
            raise Exception("You must inform a valid `mode`.")

        parameters = 0

        if mode == "SDG":
            alpha = self.settings.get('alpha', 0.01)
            iters = self.settings.get('max_iter', 100)

            parameters = _gradient_descent(df, features, label,
                                           alpha, iters, nfrag)
        elif mode == 'simple':
            """
                Simple Linear Regression: This mode is useful only if
                you have a small dataset.
            """

            xs = [[] for _ in range(nfrag)]
            ys = [[] for _ in range(nfrag)]
            xys = [[] for _ in range(nfrag)]
            for f in range(nfrag):
                xs[f] = _computation_xs(df[f], features)
                ys[f] = _computation_xs(df[f], label)
                xys[f] = _computation_XYs(df[f], features, label)

            rx = merge_reduce(_merge_calcs, xs)
            ry = merge_reduce(_merge_calcs, ys)
            rxy = merge_reduce(_merge_calcs, xys)

            parameters = _compute_line_2D(rx, ry, rxy)

        parameters = compss_wait_on(parameters)

        self.model = [parameters]
        return self

    def transform(self, data):
        """

        :param data: DDF
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        features = self.settings['feature_col']
        pred_col = self.settings['pred_col']

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _predict(df[f], features, pred_col, self.model[0])

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = \
            {'name': 'task_transform_linear_reg',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: result},
             'parent': [tmp.last_uuid],
             'output': 1, 'input': 1}

        tmp.set_n_input(uuid_key, tmp.settings['input'])
        return DDF(tmp.task_list, uuid_key)


# --------------
# Simple Linear Regression


@task(returns=list)
def _computation_xs(data, col):
    """Partial calculation."""
    sum_x = data[col].sum()
    square_x = (data[col]**2).sum()
    return [sum_x, len(data[col]), square_x]


@task(returns=list)
def _computation_XYs(xy, col1, col2):
    """Second stage of the calculation."""
    r = sum([x*y for x, y in zip(xy[col1], xy[col2])])
    return [0, 0, r]


@task(returns=list)
def _merge_calcs(p1, p2):
    """Merge calculation."""
    sum_t = p1[0] + p2[0]
    t = p1[1] + p2[1]
    square_t = p1[2] + p2[2]
    return [sum_t, t, square_t]


@task(returns=list)
def _compute_line_2D(rx, ry, rxy):
    """Generate the regression."""
    n = rx[1]
    m_x = (float(rx[0])/n)
    m_y = (float(ry[0])/n)
    b1 = float(rxy[2] - n*m_x*m_y)/(rx[2] - rx[1] * (m_x**2))
    b0 = m_y - b1*m_y

    return [b0, b1]

# --------------
# SGD mode:


def _gradient_descent(data, features, label, alpha, iters, nfrag):
    """Regression by using gradient Descent."""
    theta = np.array([0, 0, 0])

    # cost = np.zeros(iters)

    for i in range(iters):
        stage1 = [_first_stage(data[f], [features, label], theta)
                  for f in range(nfrag)]
        grad = merge_reduce(_agg_SGD, stage1)
        theta = _theta_computation(grad, alpha)

        # cost[i] = [computeCost(data[f],features,label, theta)
        #            for f in range(nfrag)]
        theta = compss_wait_on(theta)

    return theta  # ,cost


@task(returns=1)
def _first_stage(data, attr, theta):
    """Peform the partial gradient generation."""
    size = len(data)
    features, label = attr

    if size > 0:
        if isinstance(data.iloc[0][features], list):
            dim = len(data.iloc[0][features])
        else:
            dim = 1

        if (dim+1) != len(theta):
            theta = np.array([0 for _ in range(dim+1)])

        xs = np.c_[np.ones(size), np.array(data[features].tolist())]
        partial_error = np.dot(xs, theta.T) - data[label].values

        for j in range(dim+1):
            grad = np.multiply(partial_error, xs[:, j])

        return [np.sum(grad), size, dim, theta]

    return [0, 0, -1, 0]


@task(returns=list)
def _agg_SGD(error1, error2):
    """Merge the partial gradients."""
    dim1 = error1[2]
    dim2 = error2[2]

    if dim1 > 0:
        sum_grad = error1[0]+error2[0]
        size = error2[1]+error2[1]
        dim = dim1
        theta = error1[3]
    elif dim2 > 0:
        sum_grad = error1[0]+error2[0]
        size = error2[1]+error2[1]
        dim = dim2
        theta = error2[3]
    else:
        sum_grad = 0
        size = 0
        dim = -1
        theta = 0

    return [sum_grad, size, dim, theta]


@task(returns=list)
def _theta_computation(info, alpha):
    """Generate new theta."""
    grad = info[0]
    size = info[1]
    dim = info[2]
    theta = info[3]

    temp = np.zeros(theta.shape)
    for j in range(dim+1):
        temp[j] = theta[j] - ((float(alpha) / size) * grad)

    return temp


@task(returns=list)
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
                y = row[0]
                for j in xrange(1, len(row)):
                    y += row[j]*model[j]
                tmp.append(y)
        else:
            tmp = [model[0] + model[1]*row for row in data[features].values]

    data[target] = tmp
    return data