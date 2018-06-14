#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Support vector machines (SVM).

SVM is a supervised learning model used for binary classification. Given a
set of training examples, each marked as belonging to one or the other of
two categories, a SVM training algorithm builds a model that assigns new
examples to one category or the other, making it a non-probabilistic binary
linear classifier.

An SVM model is a representation of the examples as points in space, mapped
so that the examples of the separate categories are divided by a clear gap
that is as wide as possible. New examples are then mapped into that same
space and predicted to belong to a category based on which side of the gap
they fall. This algorithm is effective in high dimensional spaces and it
is still effective in cases where number of dimensions is greater than
the number of samples.

The algorithm reads a dataset composed by labels (-1.0 or 1.0) and
features (numeric fields).
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np


class SVM(object):
    """SVM's methods.

    - fit()
    - transform()
    """

    def fit(self, data, settings, nfrag):
        """Fit.

        - :param data: A list with nfrag pandas's dataframe used
               to training the model.
        - :param settings: A dictionary that contains:
         - coef_lambda: Regularization parameter (float);
         - coef_lr: Learning rate parameter (float);
         - coef_threshold: Tolerance for stopping criterion (float);
         - coef_maxIters: Number max of iterations (integer);
         - features: Fields of the features in the training data;
         - label: Fields of the labels in the training data;
        - :param nfrag: A number of fragments;
        - :return:  The model created (which is a pandas dataframe).
        """
        coef_lambda = float(settings.get('coef_lambda', 0.1))
        coef_lr = float(settings.get('coef_lr', 0.01))
        coef_threshold = float(settings.get('coef_threshold', 0.001))
        coef_maxIters = int(settings.get('coef_maxIters', 100))

        if 'features' not in settings or 'label' not in settings:
            raise Exception("You must inform the `features` and "
                            "`label` fields.")

        label = settings['label']
        features = settings['features']

        w = [0]
        old_cost = np.inf
        from pycompss.api.api import compss_wait_on
        for it in range(coef_maxIters):
            cost_grad_p = [_calc_cost_grad(data[f], f, coef_lambda,
                           w, label, features) for f in range(nfrag)]
            cost_grad = mergeReduce(_accumulate_cost_grad, cost_grad_p)
            cost_grad = compss_wait_on(cost_grad)

            cost = cost_grad[0]
            thresold = np.abs(old_cost - cost)
            if thresold <= coef_threshold:
                print "[INFO] - Final Cost %.4f" % (cost)
                break
            else:
                old_cost = cost

            w = _update_weight(coef_lr, cost_grad, w)

        model = dict()
        model['algorithm'] = 'SVM'
        model['model'] = w

        return model

    def transform(self, data, model, settings, nfrag):
        """Transform.

        :param data: A list with nfrag pandas's dataFrame 
            that will be predicted.
        :param model: A model already trained (np.array);
        :param settings: A dictionary that contains:
            - features: Field of the features in the test data;
            - predlabel: Alias to the new column with the labels predicted;
        :param nfrag: A number of fragments;
        :return: The list of dataframe with the prediction.
        """
        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        features = settings['features']
        predicted_label = settings.get('predCol', 'predited')

        if model.get('algorithm', 'null') != 'SVM':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _predict_partial(data[f], model,
                                         predicted_label, features)

        return result
    
    def transform_serial(self, data, model, settings):
        """Transform.

        :param data: A list with nfrag pandas's dataFrame 
            that will be predicted.
        :param model: A model already trained (np.array);
        :param settings: A dictionary that contains:
            - features: Field of the features in the test data;
            - predlabel: Alias to the new column with the labels predicted;
        :param nfrag: A number of fragments;
        :return: The list of dataframe with the prediction.
        """
        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        features = settings['features']
        predicted_label = settings.get('predCol', 'predited')

        if model.get('algorithm', 'null') != 'SVM':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = _predict_partial_(data, model, predicted_label, features)

        return result


# Note: If we dont use the thresold, this method must be a compss task.
# @task(returns=list)
def _update_weight(coef_lr, grad, w):
    """Update the svm's weight."""
    dim = len(grad[1])
    if dim != len(w):
        w = np.zeros(dim)

    w = np.subtract(w, np.multiply(coef_lr, grad[1]))
    return w


@task(returns=list)
def _calc_cost_grad(train_data, f, coef_lambda, w, label, features):
    """Calculate the partial cost and gradient."""
    size_train = len(train_data)
    if size_train > 0:
        dim = len(train_data.iloc[0][features])
        ypp = np.zeros(size_train)
        cost = 0
        grad = np.zeros(dim)

        if dim != len(w):
            w = [0 for _ in range(dim)]  # initial

        for i in range(size_train):
            ypp[i] = np.matmul(train_data.iloc[i][features], w)
            condition = train_data.iloc[i][label] * ypp[i]
            if (condition - 1) < 0:
                cost += (1 - condition)

        for d in range(dim):
            grad[d] = 0
            if f is 0:
                grad[d] += np.abs(coef_lambda * w[d])

            for i in range(size_train):
                i2 = train_data.iloc[i][label]
                condition = i2 * ypp[i]
                if (condition-1) < 0:
                    grad[d] -= i2 * train_data.iloc[i][features][d]

        return [cost, grad]
    else:
        return [0, 0]


@task(returns=list)
def _accumulate_cost_grad(cost_grad_p1, cost_grad_p2):
    """Merge cost and gradient."""
    cost_p1 = cost_grad_p1[0]
    cost_p2 = cost_grad_p2[0]
    grad_p1 = cost_grad_p1[1]
    grad_p2 = cost_grad_p2[1]

    cost_p1 += cost_p2
    grad_p1 = np.add(grad_p1, grad_p2)

    return [cost_p1, grad_p1]


@task(returns=list)
def _predict_partial(data, w, predicted_label, features):
    """Predict all records in a fragments."""
    return _predict_partial_(data, w, predicted_label, features)


def _predict_partial_(data, w, predicted_label, features):
    """Predict all records in a fragments."""
    if len(data) > 0:
        values = \
            np.vectorize(_predict_one, excluded=['w'])(test_xi=data[features],
                                                       w=w)
        data[predicted_label] = values.tolist()
    else:
        data[predicted_label] = np.nan

    return data


def _predict_one(test_xi, w):
    """Predict one record based in the model."""
    if np.matmul(test_xi, w) >= 0:
        return 1.0
    return -1.0
