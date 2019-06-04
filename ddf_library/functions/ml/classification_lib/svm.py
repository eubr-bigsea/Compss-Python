#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF
from ddf_library.utils import generate_info
from ddf_library.ddf_model import ModelDDF

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on

import numpy as np


class SVM(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    Support vector machines (SVM) is a supervised learning model used for
    binary classification. Given a set of training examples, each marked as
    belonging to one or the other of two categories, a SVM training algorithm
    builds a model that assigns new examples to one category or the other,
    making it a non-probabilistic binary linear classifier.

    An SVM model is a representation of the examples as points in space, mapped
    so that the examples of the separate categories are divided by a clear gap
    that is as wide as possible. New examples are then mapped into that same
    space and predicted to belong to a category based on which side of the gap
    they fall. This algorithm is effective in high dimensional spaces and it
    is still effective in cases where number of dimensions is greater than
    the number of samples.

    The algorithm reads a data set composed by labels (-1 or 1) and
    features (numeric fields).

    :Example:

    >>> cls = SVM(feature_col='features', label_col='label',
    >>>           max_iter=10).fit(ddf1)
    >>> ddf2 = cls.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, coef_lambda=0.1,
                 coef_lr=0.01, threshold=0.001, max_iter=100, penalty='l2'):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param coef_lambda: Regularization parameter (default, 0.1);
        :param coef_lr: Learning rate parameter (default, 0.1);
        :param threshold: Tolerance for stopping criterion (default, 0.001);
        :param max_iter: Number max of iterations (default, 100);
        :param penalty: Apply 'l2' or 'l1' penalization (default, 'l2')
        """
        super(SVM, self).__init__()

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['coef_lambda'] = coef_lambda
        self.settings['coef_lr'] = coef_lr
        self.settings['threshold'] = threshold
        self.settings['max_iter'] = max_iter
        self.settings['regularization'] = penalty

        self.model = {}
        self.name = 'SVM'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        coef_lambda = float(self.settings.get('coef_lambda', 0.1))
        coef_lr = float(self.settings.get('coef_lr', 0.01))
        coef_threshold = float(self.settings.get('threshold', 0.001))
        coef_max_iter = int(self.settings.get('max_iter', 100))
        regularization = self.settings.get('regularization', 'l2')
        df, nfrag, tmp = self._ddf_initial_setup(data)

        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']

        w = np.zeros(1, dtype=float)
        old_cost = np.inf
        old_w = w.copy()

        cost_grad_p = [[] for _ in range(nfrag)]

        for it in range(coef_max_iter):
            if it == 0:
                for f in range(nfrag):
                    cost_grad_p[f], df[f] = \
                        _calc_cost_grad_first(df[f], w, col_label, col_feature)
            else:
                for f in range(nfrag):
                    cost_grad_p[f] = _calc_cost_grad(df[f], w)

            cost_grad = merge_reduce(_accumulate_cost_grad, cost_grad_p)

            w, cost = _update_weight(coef_lr, cost_grad, w,
                                     coef_lambda, regularization)

            print("[INFO] SVM - it {} - cost:{:.4f}".format(it, cost))
            threshold = old_cost - cost
            if threshold <= coef_threshold:
                w = old_w
                break
            else:
                old_cost = cost
                old_w = w

        self.model['model'] = w
        self.model['algorithm'] = self.name

        return self

    def fit_transform(self, data, pred_col='prediction_SVM'):
        """
        Fit the model and transform.

        :param data: DDF
        :param pred_col: Output prediction name (default, *'prediction_SVM'*);
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, pred_col)

        return ddf

    def transform(self, data, pred_col='prediction_SVM'):
        """

        :param data: DDF
        :param pred_col: Output prediction name (default, *'prediction_SVM'*);
        :return: DDF
        """

        self.check_fitted_model()

        task_list = data.task_list
        settings = self.settings.copy()
        settings['pred_col'] = pred_col
        settings['model'] = self.model['model'].copy()

        def task_transform_svm(df, params):
            return _svm_predict(df, params)

        uuid_key = self._ddf_add_task(task_name='task_transform_svm',
                                      opt=self.OPT_SERIAL,
                                      function=[task_transform_svm,
                                                settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=task_list, last_uuid=uuid_key)


def _update_weight(coef_lr, cost_grad, w, coef_lambda, regularization):
    """Update the svm's weight."""
    loss, grad, size = compss_wait_on(cost_grad)

    dim = len(grad)
    if dim != len(w):
        w = np.zeros(dim)

    penalty = 0
    if regularization == 'l1':  # Lasso
        # Lasso adds absolute value of magnitude of coefficient as penalty term
        penalty = coef_lambda * np.linalg.norm(w, 1)

    elif regularization == 'l2':  # Ridge
        # Ridge adds squared magnitude of coefficient as penalty term
        penalty = coef_lambda * (np.linalg.norm(w))**2

    dw = (grad/size) + 2 * penalty

    loss = loss/size + penalty

    w = w - coef_lr * dw

    return w, loss


@task(returns=2)
def _calc_cost_grad_first(train_data, w, label, features):
    """Calculate the partial cost and gradient."""
    size_train = train_data.shape[0]
    labels = train_data[label].values
    train_data = train_data[features].values

    if size_train > 0:

        dim = train_data.shape[1]
        if dim != len(w):
            w = np.zeros(dim, dtype=float)  # initial

        prediction = (labels * np.dot(train_data, w))

        # hinge loss (select negative values)
        idx = np.nonzero((prediction - 1) < 0)
        loss = np.sum(1 - prediction[idx])

        # -y * x for all values lesser than 1
        grad = - np.dot(labels[idx], train_data[idx])

        return [loss, grad, size_train], [labels, train_data]
    else:
        return [0, 0, size_train], [labels, train_data]


@task(returns=1)
def _calc_cost_grad(train_data, w):
    """Calculate the partial cost and gradient."""
    labels, train_data = train_data
    size_train = train_data.shape[0]

    if size_train > 0:

        dim = train_data.shape[1]
        if dim != len(w):
            w = np.zeros(dim, dtype=float)  # initial

        prediction = (labels * np.dot(train_data, w))

        # hinge loss (select negative values)
        idx = np.nonzero((prediction - 1) < 0)
        loss = np.sum(1 - prediction[idx])

        # -y * x for all values lesser than 1
        grad = - np.dot(labels[idx], train_data[idx])

        return [loss, grad, size_train]
    else:
        return [0, 0, size_train]


@task(returns=1)
def _accumulate_cost_grad(cost_grad_p1, cost_grad_p2):
    """Merge cost and gradient."""
    cost_p1, grad_p1, size_p1 = cost_grad_p1
    cost_p2, grad_p2, size_p2 = cost_grad_p2

    cost_p1 += cost_p2
    size_p1 += size_p2
    grad_p1 = np.add(grad_p1, grad_p2)

    return [cost_p1, grad_p1, size_p1]


def _svm_predict(data, settings):
    """Predict all records in a fragments."""

    pred_col = settings['pred_col']
    features = settings['feature_col']
    frag = settings['id_frag']
    w = settings['model']

    # TODO: add different classes

    if pred_col in data.columns:
        data.drop([pred_col], axis=1, inplace=True)

    if len(data) > 0:
        # values = np.matmul(data[features].values, w)
        # values = np.where(values >= 0, 1, -1)
        values = np.sign(np.dot(data[features].values, w)).astype(int)
        data[pred_col] = values
    else:
        data[pred_col] = np.nan

    info = generate_info(data, frag)
    return data, info
