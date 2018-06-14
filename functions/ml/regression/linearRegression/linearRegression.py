#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""Linear regression.

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
from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
import numpy as np


class linearRegression(object):
    """Linear Regression's Methods.

    - fit()
    - transform()
    """

    def fit(self, data, settings, nfrag):
        """Fit.

        - :param data:      A list with nfrag pandas's dataframe
                            used to create the model.
        - :param settings:  A dictionary that contains:
            - features: 	Field of the features in the dataset;
            - label: 	    Field of the label in the dataset;
            - mode:
                * 'simple': Best option if is a 2D regression;
                * 'SDG':    Uses a Stochastic gradient descent to perform
                            the regression. Can be used to data of all
                            dimensions.
            - max_iter:     Maximum number of iterations, only using 'SDG'
                            (integer, default: 100);
            - alpha:        Learning rate parameter, only using 'SDG'
                            (float, default 0.01)
        - :param nfrag:   A number of fragments;
        - :return:          Returns a model (which is a pandas dataframe).

        Note: Best results with a normalizated data.
        """
        features = settings['features']
        label = settings['label']
        mode = settings.get('mode', 'SDG')

        if mode not in ['simple', 'SDG']:
            raise Exception("You must inform a valid `mode`.")

        if mode == "SDG":
            alpha = settings.get('alpha', 0.1)
            iters = settings.get('max_iter', 100)

            parameters = _gradient_descent(data, features, label,
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
                xs[f] = _computation_xs(data[f], features)
                ys[f] = _computation_xs(data[f], label)
                xys[f] = _computation_XYs(data[f], features, label)

            rx = mergeReduce(_merge_calcs, xs)
            ry = mergeReduce(_merge_calcs, ys)
            rxy = mergeReduce(_merge_calcs, xys)

            parameters = _compute_line_2D(rx, ry, rxy)

        from pycompss.api.api import compss_wait_on
        parameters = compss_wait_on(parameters)
        model = dict()
        model['algorithm'] = 'linearRegression'
        model['model'] = parameters
        return model

    def transform(self, data, model, settings, nfrag):
        """Transform.

        - :param data:      A list with nfrag pandas's dataframe
                            that will be _predicted.
        - :param model:		The Linear Regression's model created;
        - :param settings:  A dictionary that contains:
            - features: 	Field of the features in the test data;
            - predCol:    	Alias to the new _predicted labels;
        - :param nfrag:   A number of fragments;
        - :return:          The prediction (in the same input format).
        """
        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        features = settings['features']
        pred_col = settings.get('predCol', 'predicted_value')

        if model.get('algorithm', 'null') != 'linearRegression':
            raise Exception("You must inform a valid Linear Regression model.")

        model = model['model']
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _predict(data[f], features, pred_col, model)

        return result

    def transform_serial(self, data, model, settings):
        """Transform.

                - :param data:      A list with nfrag pandas's dataframe
                                    that will be _predicted.
                - :param model:		The Linear Regression's model created;
                - :param settings:  A dictionary that contains:
                    - features: 	Field of the features in the test data;
                    - predCol:    	Alias to the new _predicted labels;
                - :param nfrag:   A number of fragments;
                - :return:          The _prediction (in the same input format).
                """
        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        features = settings['features']
        pred_col = settings.get('predCol', 'predicted_value')

        if model.get('algorithm', 'null') != 'linearRegression':
            raise Exception("You must inform a valid Linear Regression model.")

        model = model['model']
        result = _predict_(data, features, pred_col, model)

        return result


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
        stage1 = [_first_stage(data[f], features, label, theta)
                  for f in range(nfrag)]
        grad = mergeReduce(_agg_SGD, stage1)
        theta = _theta_computation(grad, alpha)

        # cost[i] = [computeCost(data[f],features,label, theta)
        #            for f in range(nfrag)]
        theta = compss_wait_on(theta)

    return theta  # ,cost


@task(returns=list)
def _first_stage(data, features, target, theta):
    """Peform the partial gradient generation."""
    size = len(data)

    if size > 0:
        if isinstance(data.iloc[0][features], list):
            dim = len(data.iloc[0][features])
        else:
            dim = 1

        if (dim+1) != len(theta):
            theta = np.array([0 for _ in range(dim+1)])

        xs = np.c_[np.ones(size), np.array(data[features].tolist())]
        partial_error = np.dot(xs, theta.T) - data[target].values

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
