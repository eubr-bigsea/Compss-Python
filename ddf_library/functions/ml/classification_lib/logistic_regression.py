#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info, read_stage_file
from ddf_library.bases.ddf_model import ModelDDF

from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on, compss_delete_object

import numpy as np


class LogisticRegression(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Logistic regression is named for the function used at the core
    of the method, the logistic function. It is the go-to method for
    binary classification problems (problems with two class values).

    The logistic function, also called the sigmoid function was
    developed by statisticians to describe properties of population
    growth in ecology, rising quickly and maxing out at the carrying
    capacity of the environment. It’s an S-shaped curve that can take
    any real-valued number and map it into a value between 0 and 1,
    but never exactly at those limits.

    This implementation uses a Gradient Ascent (a variant of
    the gradient descent). Gradient ascent is the same as gradient
    descent, except I’m maximizing instead of minimizing a function.

    :Example:

    >>> cls = LogisticRegression()\
    >>>            .fit(ddf1, feature_col='features', label_col='label')
    >>> ddf2 = cls.transform(ddf1)
    """

    def __init__(self, alpha=0.1, regularization=0.1, max_iter=100,
                 threshold=0.01):
        """
        :param alpha: Learning rate parameter (default, 0.1);
        :param regularization: Regularization parameter (default, 0.1);
        :param max_iter: Maximum number of iterations (default, 100);
        :param threshold: Tolerance for stopping criterion (default, 0.01);
        """
        super(LogisticRegression, self).__init__()

        self.alpha = alpha
        self.regularization = regularization
        self.threshold = threshold
        self.max_iter = max_iter

        self.feature_col = None
        self.label_col = None
        self.pred_col = None

    def fit(self, data, feature_col, label_col):
        """
        Fit the model.

        :param data: DDF
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        self.feature_col = feature_col
        self.label_col = label_col

        parameters = _logr_compute_theta(df, self.feature_col, self.label_col,
                                         self.alpha, self.max_iter,
                                         self.threshold, self.regularization,
                                         nfrag)

        self.model['model'] = compss_wait_on(parameters)
        self.model['algorithm'] = self.name
        return self

    def fit_transform(self, data, feature_col, label_col,
                      pred_col='prediction_LogReg'):
        """
        Fit the model and transform.

        :param data: DDF
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param pred_col: Output prediction name (default,
         *'prediction_LogReg'*);
        :return: DDF
        """

        self.fit(data, feature_col, label_col)
        ddf = self.transform(data, pred_col=pred_col)

        return ddf

    def transform(self, data, feature_col=None, pred_col='prediction_LogReg'):
        """

        :param data: DDF
        :param feature_col: Feature column name;
        :param pred_col: Output prediction name (default,
         *'prediction_LogReg'*);
        :return: DDF
        """

        self.check_fitted_model()
        if feature_col:
            self.feature_col = feature_col
        self.pred_col = pred_col

        settings = self.__dict__.copy()
        settings['model'] = settings['model']['model']

        def task_transform_logr(df, params):
            return _logr_predict(df, params)

        uuid_key = self._ddf_add_task(task_name='task_transform_logr',
                                      opt=self.OPT_SERIAL,
                                      function=[task_transform_logr,
                                                settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=data.task_list, last_uuid=uuid_key)


def _logr_sigmoid(scores):
    """
    Evaluate the sigmoid function to scores.
    :param scores:
    :return: Value returned.
    """
    return 1 / (1 + np.exp(-scores))


def _logr_compute_theta(data, features, label, alpha, max_iter,
                        threshold, reg, nfrag):
    """
    Perform a logistic regression via Gradient Ascent
    """

    theta = np.zeros(1, dtype=float)  # initial
    it = 0
    converged = False

    xy = [[] for _ in range(nfrag)]
    gr_in = [[] for _ in range(nfrag)]

    while (it < max_iter) and not converged:
        # grEin = gradient of in-sample Error
        if it == 0:
            for f in range(nfrag):
                xy[f], gr_in[f] = _logr_ga_init(data[f], features, label, theta)
        else:
            for f in range(nfrag):
                gr_in[f] = _logr_gradient_ascent(xy[f], theta)

        grad = merge_reduce(_logr_agg_sga, gr_in)
        result = _logr_calc_theta(grad, alpha, it, reg, threshold)
        it += 1
        theta, converged = result

    compss_delete_object(xy)

    return theta


@task(returns=2, data_input=FILE_IN)
def _logr_ga_init(data_input, feature_col, label_col, theta):
    """
    Estimate logistic regression coefficients using Gradient Ascent.
    """
    data = read_stage_file(data_input, feature_col + [label_col])
    size = len(data)
    if size == 0:
        return [[], 0, 0, theta]

    # similar to np.concatenate
    features = np.c_[np.ones(size), data[feature_col].values]
    target = data[label_col].values
    del data

    dim = features.shape[1]

    if dim != len(theta):
        theta = np.array(np.zeros(dim), dtype=float)

    # get the sum of error
    scores = np.dot(features, theta)
    predictions = _logr_sigmoid(scores)

    # Update weights with gradient
    output_error_signal = target - predictions
    gradient = np.dot(features.T, output_error_signal)

    return [features, target], [gradient, size, dim, theta]


@task(returns=1)
def _logr_gradient_ascent(data, theta):
    """
    Estimate logistic regression coefficients using Gradient Ascent.
    """

    size = len(data)
    dim = len(theta)
    if size == 0:
        return [[], 0, 0, theta]

    features, target = data

    # get the sum of error
    scores = np.dot(features, theta)
    predictions = _logr_sigmoid(scores)

    # Update weights with gradient
    output_error_signal = target - predictions
    gradient = np.dot(features.T, output_error_signal)

    return [gradient, size, dim, theta]


@task(returns=1)
def _logr_agg_sga(info1, info2):

    if len(info1[0]) > 0:
        if len(info2[0]) > 0:
            gradient = info1[0]+info2[0]
        else:
            gradient = info1[0]
    else:
        gradient = info2[0]

    size = info2[1] + info2[1]
    dim = info1[2] if info1[2] != 0 else info2[2]
    theta = info1[3] if len(info1[3]) > len(info2[3]) else info2[3]

    return [gradient, size, dim, theta]


def _logr_calc_theta(info, coef_lr, it, regularization, threshold):
    info = compss_wait_on(info)
    gradient = info[0]
    theta = info[3]

    # update coefficients
    alpha = coef_lr/(1+it)
    theta += coef_lr*(gradient - regularization*theta)

    converged = False
    cost = alpha * np.sum(gradient*gradient)
    print("[INFO] Logistic Regressing - it {} - cost:{:.4f}".format(it, cost))
    if cost <= threshold:
        converged = True
    return [theta, converged]


def _logr_predict(data, settings):

    frag = settings['id_frag']
    col_features = settings['feature_col']
    pred_col = settings['pred_col']
    theta = settings['model']
    size = len(data)

    if pred_col in data.columns:
        data.drop([pred_col], axis=1, inplace=True)

    if len(data) > 0:

        xs = np.c_[np.ones(size), data[col_features].values]
        xs = np.dot(xs, theta)
        data[pred_col] = np.rint(_logr_sigmoid(xs)).astype(int)
    else:
        data[pred_col] = np.nan

    info = generate_info(data, frag)
    return data, info
