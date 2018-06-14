#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import math
import numpy as np


class logisticRegression(object):

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

    This implementation uses a Stochastic Gradient Ascent (a variant of
    the Stochastic gradient descent). It is called stochastic because
    the derivative based on a randomly chosen single example is a random
    approximation to the true derivative based on all the training data.

    Methods:
        - fit()
        - transform()

    """

    def fit(self, data, settings, nfrag):

        """
        fit():

        :param data: A list with nfrag pandas's dataframe used to
                training the model.
        :param settings: A dictionary that contains:
            - iters: Maximum number of iterations (integer, default is 100);
            - threshold: Tolerance for stopping criterion
                (float, default is 0.001);
            - regularization: Regularization parameter (float, default is 0.1);
            - alpha: The Learning rate, it means, how large of steps to take
                on our cost curve (float, default is 0.1);
            - features: Field of the features in the training data;
            - label: Field of the labels in the training data;
        :param nfrag: A number of fragments;
        :return: The model created (which is a pandas dataframe).
        """

        if 'features' not in settings or 'label' not in settings:
            raise Exception("You must inform the `features` "
                            "and `label` fields.")

        features = settings['features']
        label = settings['label']
        alpha = settings.get('alpha', 0.1)
        reg = settings.get('regularization', 0.1)
        iters = settings.get('iters', 100)
        threshold = settings.get('threshold', 0.001)

        parameters = _compute_coeffs(data, features, label, alpha,
                                     iters, threshold, reg, nfrag)

        model = dict()
        model['algorithm'] = 'logisticRegression'
        model['model'] = parameters

        return model

    def transform(self, data, model, settings, nfrag):
        """
        transform():

        :param data: A list with nfrag pandas's dataframe that
             will be predicted.
        :param model: The Logistic Regression model created;
        :param settings: A dictionary that contains:
          - features: Field of the features in the test data;
          - predCol: Alias to the new column with the labels predicted;
        :param nfrag: A number of fragments;
        :return: The prediction (in the same input format).
        """
        if 'features' not in settings:
            raise Exception("You must inform the `features`  field.")

        if model.get('algorithm', 'null') != 'logisticRegression':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _predict(data[f], settings, model)

        return result

    def transform_serial(self, data, model, settings):
        """
        transform():

        :param data: A list with nfrag pandas's dataframe that
             will be predicted.
        :param model: The Logistic Regression model created;
        :param settings: A dictionary that contains:
          - features: Field of the features in the test data;
          - predCol: Alias to the new column with the labels predicted;
        :param nfrag: A number of fragments;
        :return: The prediction (in the same input format).
        """
        if 'features' not in settings:
            raise Exception("You must inform the `features`  field.")

        if model.get('algorithm', 'null') != 'logisticRegression':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = _predict_(data, settings, model)

        return result


def _sigmoid(x, w):
    """
    Evaluate the sigmoid function at x.
    :param x: Vector.
    :param w: weight
    :return: Value returned.
    """
    try:
        den = math.exp(sum(w*x))
    except OverflowError:
        den = float('inf')
    return 1.0 - 1.0/(1.0 + den)


def _compute_coeffs(data, features, label, alpha, iters,
                    threshold, reg, nfrag):
    """
    Perform a logistic regression via gradient ascent.
    """
    from pycompss.api.api import compss_wait_on

    theta = np.array(np.zeros(1), dtype=float)  # initial
    i = reg = 0
    converged = False
    while (i < iters) and not converged:
        # grEin = gradient of in-sample Error
        gr_in = [_gradient_ascent(data[f], features, label, theta, alpha)
                 for f in range(nfrag)]
        grad = mergeReduce(_agg_sga, gr_in)
        result = _calcTheta(grad, alpha, i, reg, threshold)
        result = compss_wait_on(result)
        i += 1
        theta, converged = result

    theta = compss_wait_on(theta)

    return theta


@task(returns=list)
def _gradient_ascent(data, X, Y, theta, alfa):
    """
        Estimate logistic regression coefficients
        using stochastic gradient descent.
    """

    if len(data) == 0:
        return [[], 0, 0, theta]

    dim = len(data.iloc[0][X])

    if (dim+1) != len(theta):
        theta = np.array(np.zeros(dim+1), dtype=float)

    size = len(data)
    # get the sum of error
    gradient = 0

    xs = np.c_[np.ones(size), np.array(data[X].tolist())]  # adding ones

    for n in range(size):
        xn = np.array(xs[n, :])
        yn = data[Y].values[n]
        grad_p = _sigmoid(xn, theta)
        gradient += xn*(yn - grad_p)

    return [gradient, size, dim, theta]


@task(returns=list)
def _agg_sga(info1, info2):

    if len(info1[0]) > 0:
        if len(info2[0]) > 0:
            gradient = info1[0]+info2[0]
        else:
            gradient = info1[0]
    else:
        gradient = info2[0]

    size = info2[1]+info2[1]
    dim = info1[2] if info1[2] != 0 else info2[2]
    theta = info1[3] if len(info1[3]) > len(info2[3]) else info2[3]

    return [gradient, size, dim, theta]


@task(returns=list)
def _calcTheta(info, coef_lr, it, regularization, threshold):
    gradient = info[0]
    theta = info[3]

    # update coefficients
    alpha = coef_lr/(1+it)
    theta += alpha*(gradient - regularization*theta)

    converged = False
    if alpha*sum(gradient*gradient) < threshold:
        converged = True
    return [theta, converged]


@task(returns=list)
def _predict(data, settings, theta):
    return _predict_(data, settings, theta)


def _predict_(data, settings, theta):
    col_features = settings['features']
    pred_col = settings.get('predCol', 'prediction')
    size = len(data)

    xs = np.c_[np.ones(size), np.array(data[col_features].tolist())]
    data[pred_col] = [round(_sigmoid(x, theta)) for x in xs]
    return data
