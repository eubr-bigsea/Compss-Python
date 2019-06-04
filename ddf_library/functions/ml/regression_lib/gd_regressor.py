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


class GDRegressor(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Linear model fitted by minimizing a regularized empirical loss with
    Gradient Descent.

    :Example:

    >>> model = GDRegressor('features', 'y').fit(ddf1)
    >>> ddf2 = model.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, max_iter=100, alpha=1, tol=1e-3):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param max_iter: Maximum number of iterations (default, 100);
        :param alpha: learning rate parameter  (default, 1). This method sets
         the learning rate parameter used by Gradient Descent when updating
         the hypothesis after each iteration. Up to a point, higher values will
         cause the algorithm to converge on the optimal solution more quickly,
         however if the value is set too high then it will fail to converge at
         all, yielding successively larger errors on each iteration;
        :param tol: Tolerance stop criteria (default, 1e-3).
        """
        super(GDRegressor, self).__init__()

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['max_iter'] = max_iter
        self.settings['alpha'] = alpha
        self.settings['tolerance'] = tol

        self.model = {}
        self.name = 'GDRegressor'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        features = self.settings['feature_col']
        label = self.settings['label_col']
        alpha = self.settings['alpha']
        max_iter = self.settings['max_iter']
        tol = self.settings['tolerance']

        parameters = _gradient_descent(df, features, label,
                                       alpha, max_iter, tol, nfrag)

        parameters = compss_wait_on(parameters)
        self.model['model'] = parameters
        self.model['algorithm'] = self.name
        return self

    def fit_transform(self, data, pred_col='pred_LinearReg'):
        """
        Fit the model and transform.

        :param data: DDF
        :param pred_col: Output prediction column (default, *'pred_LinearReg'*);
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, pred_col=pred_col)

        return ddf

    def transform(self, data, pred_col='pred_LinearReg'):
        """
        :param data: DDF
        :param pred_col: Output prediction column (default, *'pred_LinearReg'*);
        :return: DDF
        """

        self.check_fitted_model()

        task_list = data.task_list
        settings = self.settings.copy()
        settings['pred_col'] = pred_col
        settings['model'] = self.model['model'].copy()

        def task_gd_regressor(df, params):
            return _predict(df, params)

        uuid_key = self._ddf_add_task(task_name='task_gd_regressor',
                                      opt=self.OPT_SERIAL,
                                      function=[task_gd_regressor, settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=task_list, last_uuid=uuid_key)


def _gradient_descent(data, features, label, alpha, max_iter, tol, nfrag):
    """Regression by using gradient Descent."""
    theta = np.random.uniform(size=3)

    input_data = [_select_att(data[f], [features, label]) for f in range(nfrag)]

    previous_loss = np.inf

    for it in range(max_iter):
        stage1 = [_lr_gb_first_stage(input_data[f], theta)
                  for f in range(nfrag)]
        grad = merge_reduce(_lr_gb_agg, stage1)

        theta, loss = _lr_gb_theta_computation(grad, alpha)

        compss_delete_object(stage1)

        print("[INFO] GD Regressor - it {} - loss:{:.4f}".format(it, loss))

        if np.abs(loss - previous_loss) <= tol:
            break

    return theta


@task(returns=1)
def _select_att(data, cols):
    features, label = cols
    to_keep = features + [label]
    data = data[to_keep].dropna()

    features = data[features].values
    labels = data[label].values
    del data
    # Translates slice objects to concatenation along the second axis.
    features = np.c_[np.ones(len(labels)), features]
    return [features, labels]


@task(returns=1)
def _lr_gb_first_stage(data, theta):
    """Perform the partial gradient generation."""

    features, label = data

    size = len(label)

    if size > 0:

        dim = features.shape[1]

        if dim != len(theta):
            theta = np.random.uniform(size=dim)

        partial_error = np.dot(features, theta) - label

        gradient = np.dot(partial_error.T, features)

        sse = np.sum(partial_error ** 2)

        return [gradient, size, dim, theta, sse]

    return [0, 0, -1, 0, 0]


@task(returns=1)
def _lr_gb_agg(error1, error2):
    """Merge the partial gradients."""
    grad1, size1, dim1, theta1, sse1 = error1
    grad2, size2, dim2, theta2, sse2 = error2

    theta = theta1
    dim = dim1
    if dim2 > dim1:
        # meaning that partition1 is empty
        dim = dim2
        theta = theta2

    sum_grad = np.sum([grad1, grad2], axis=0)
    size = size1 + size2
    sse = sse1 + sse2

    return [sum_grad, size, dim, theta, sse]


def _lr_gb_theta_computation(info, alpha):
    """Generate new theta."""
    info = compss_wait_on(info)
    grad, size, dim, theta, sse = info

    theta -= (alpha / size) * grad

    loss = sse / (2 * size)

    return theta, loss


def _predict(data, settings):
    """Predict the values."""
    n_rows = len(data)
    frag = settings['id_frag']
    model = settings['model']
    pred_col = settings['pred_col']
    features = settings['feature_col']

    if len(data) > 0:

        xs = np.c_[np.ones(n_rows), data[features].values]
        tmp = np.dot(xs, model)

    else:
        tmp = np.nan

    if pred_col in data.columns:
        data.drop([pred_col], axis=1, inplace=True)

    data[pred_col] = tmp

    info = generate_info(data, frag)
    return data, info
