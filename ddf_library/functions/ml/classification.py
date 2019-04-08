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
from pycompss.api.local import *
from ddf_library.ddf import DDF
from ddf_library.ddf_model import ModelDDF

__all__ = ['KNearestNeighbors', 'GaussianNB', 'LogisticRegression', 'SVM']


import sys
sys.path.append('../../')


class KNearestNeighbors(ModelDDF):

    """K-Nearest Neighbor is a algorithm used that can be used for both
    classification and regression predictive problems. However, it is more
    widely used in classification problems. In a classification, the
    algorithm computes from a simple majority vote of the K nearest neighbors
    of each point present in the training set. The choice of the parameter
    K is very crucial in this algorithm, and depends on the dataset.
    However, values of one or tree is more commom.

    :Example:

    >>> knn = KNearestNeighbors(feature_col='features',
    >>>                         label_col='label', k=1).fit(ddf1)
    >>> ddf2 = knn.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, pred_col=None, k=3):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param pred_col: Output prediction name (default, *'prediction_kNN'*);
        :param k: Number of nearest neighbors to majority vote;
        """
        super(KNearestNeighbors, self).__init__()

        if not pred_col:
            pred_col = 'prediction_kNN'

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['pred_col'] = pred_col
        self.settings['k'] = k

        self.model = []
        self.name = 'KNearestNeighbors'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']
        k = self.settings['k']

        train_data = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            train_data[f] = _knn_create_model(df[f], col_label,
                                              col_feature, nfrag, k)
        model = merge_reduce(merge_lists, train_data)

        self.model = [compss_wait_on(model)]
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

    def transform(self, data, feature_col=None):
        """

        :param data: DDF
        :param feature_col: Feature column name
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        self.settings['model'] = self.model[0]

        if feature_col is not None:
            self.settings['feature_col'] = feature_col

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            result[i], info[i] = _knn_classify_block_(df[i], self.settings)

        uuid_key = self._ddf_add_task(task_name='task_transform_knn',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=1)
def _knn_create_model(df, label, features, nfrag, k):
    """Create a partial model based in the selected columns."""
    labels = df[label].values
    feature = np.array(df[features].values.tolist())
    return [labels, feature, 0, nfrag, k]


@task(returns=1)
def merge_lists(list1, list2):
    """Merge all elements in an unique dataframe to be part of a knn model."""
    l1, f1, i1, nfrag, k = list1
    l2, f2, i2, _, _ = list2

    if len(l1) != 0 and len(l2) != 0:
        l1 = np.concatenate((l1, l2), axis=0)
        f1 = np.concatenate((f1, f2), axis=0)

    elif len(l1) < len(l2):
        l1, f1 = l2, f2

    i = i1+i2 + 1
    if i == (nfrag - 1):
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(f1, l1)
        return [neigh]

    return [l1, f1, i, nfrag, k]


@task(returns=2)
def _knn_classify_block_(data, settings):
    """Perform a partial classification."""
    col_features = settings['feature_col']
    pred_col = settings.get('pred_col', "prediction_kNN")
    model = settings['model'][0]

    if len(data) == 0:
        data[pred_col] = np.nan
        info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
        return data, info

    values = model.predict(data[col_features].tolist())
    data[pred_col] = values.tolist()

    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
    return data, info


class SVM(ModelDDF):
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

    The algorithm reads a dataset composed by labels (-1 or 1) and
    features (numeric fields).

    :Example:

    >>> svm = SVM(feature_col='features', label_col='label',
    >>>           max_iters=10).fit(ddf1)
    >>> ddf2 = svm.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, pred_col=None,
                 coef_lambda=0.1, coef_lr=0.01, threshold=0.001, max_iters=100):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param pred_col: Output prediction name (default, *'prediction_SVM'*);
        :param coef_lambda: Regularization parameter (default, 0.1);
        :param coef_lr: Learning rate parameter (default, 0.1);
        :param threshold: Tolerance for stopping criterion (default, 0.001);
        :param max_iters: Number max of iterations (default, 100).
        """
        super(SVM, self).__init__()

        if not pred_col:
            pred_col = 'prediction_SVM'

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['pred_col'] = pred_col
        self.settings['coef_lambda'] = coef_lambda
        self.settings['coef_lr'] = coef_lr
        self.settings['threshold'] = threshold
        self.settings['maxIters'] = max_iters

        self.model = []
        self.name = 'SVM'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        coef_lambda = float(self.settings.get('coef_lambda', 0.1))
        coef_lr = float(self.settings.get('coef_lr', 0.01))
        coef_threshold = float(self.settings.get('coef_threshold', 0.001))
        coef_max_iter = int(self.settings.get('coef_maxIters', 100))

        df, nfrag, tmp = self._ddf_inital_setup(data)

        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']

        w = [0]
        old_cost = np.inf

        cost_grad_p = [[] for _ in range(nfrag)]

        for it in range(coef_max_iter):
            if it == 0:
                for f in range(nfrag):
                    cost_grad_p[f], df[f] = \
                        _calc_cost_grad_first(df[f], f, coef_lambda, w, 
                                              col_label, col_feature)
            else:
                for f in range(nfrag):
                    cost_grad_p[f] = _calc_cost_grad(df[f], f, coef_lambda, w) 

            cost_grad = merge_reduce(_accumulate_cost_grad, cost_grad_p)
            cost_grad = compss_wait_on(cost_grad)

            cost = cost_grad[0]
            thresold = np.abs(old_cost - cost)
            if thresold <= coef_threshold:
                break
            else:
                old_cost = cost

            w = _update_weight(coef_lr, cost_grad, w)
        print "[INFO] - Final Cost %.4f" % cost

        self.model = [w]

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

        cols = [self.settings['feature_col'], self.settings['pred_col']]

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _predict_partial(df[f], self.model[0], cols)

        uuid_key = self._ddf_add_task(task_name='task_transform_svm',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@local
def _update_weight(coef_lr, grad, w):
    """Update the svm's weight."""
    dim = len(grad[1])
    if dim != len(w):
        w = np.zeros(dim)

    w = np.subtract(w, np.multiply(coef_lr, grad[1]))
    return w


@task(returns=2)
def _calc_cost_grad_first(train_data, f, coef_lambda, w, label, features):
    """Calculate the partial cost and gradient."""
    size_train = len(train_data)
    labels = train_data[label].values
    train_data = train_data[features].values

    if size_train > 0:

        dim = len(train_data[0])
        ypp, grad, cost = np.zeros(size_train), np.zeros(dim), 0

        if dim != len(w):
            w = [0 for _ in range(dim)]  # initial

        for i in range(size_train):
            ypp[i] = np.matmul(train_data[i], w)
            condition = labels[i] * ypp[i]
            if (condition - 1) < 0:
                cost += (1 - condition)

        for d in range(dim):
            grad[d] = 0
            if f is 0:
                grad[d] += np.abs(coef_lambda * w[d])

            for i in range(size_train):
                i2 = labels[i]
                condition = i2 * ypp[i]
                if (condition - 1) < 0:
                    grad[d] -= i2 * train_data[i][d]

        return [cost, grad], [labels, train_data]
    else:
        return [0, 0], [labels, train_data]


@task(returns=1)
def _calc_cost_grad(train_data, f, coef_lambda, w):
    """Calculate the partial cost and gradient."""
    labels, train_data = train_data
    size_train = len(train_data)
    
    if size_train > 0:

        dim = len(train_data[0])
        ypp, grad, cost = np.zeros(size_train), np.zeros(dim), 0

        for i in range(size_train):
            ypp[i] = np.matmul(train_data[i], w)
            condition = labels[i] * ypp[i]
            if (condition - 1) < 0:
                cost += (1 - condition)

        for d in range(dim):
            grad[d] = 0
            if f is 0:
                grad[d] += np.abs(coef_lambda * w[d])

            for i in range(size_train):
                i2 = labels[i]
                condition = i2 * ypp[i]
                if (condition - 1) < 0:
                    grad[d] -= i2 * train_data[i][d]

        return [cost, grad]
    else:
        return [0, 0]
    
    
@task(returns=1)
def _accumulate_cost_grad(cost_grad_p1, cost_grad_p2):
    """Merge cost and gradient."""
    cost_p1, grad_p1 = cost_grad_p1
    cost_p2, grad_p2 = cost_grad_p2

    cost_p1 += cost_p2
    grad_p1 = np.add(grad_p1, grad_p2)

    return [cost_p1, grad_p1]


@task(returns=2)
def _predict_partial(data, w, cols):
    """Predict all records in a fragments."""
    features, predicted_label = cols

    if len(data) > 0:
        values = np.matmul(data[features].tolist(), w)
        values = np.where(values >= 0, 1, -1)
        data[predicted_label] = values.tolist()
    else:
        data[predicted_label] = np.nan

    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
    return data, info


class LogisticRegression(ModelDDF):

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

    :Example:

    >>> logr = LogisticRegression(feature_col='features',
    >>>                           label_col='label').fit(ddf1)
    >>> ddf2 = logr.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, pred_col=None, alpha=0.1,
                 regularization=0.1, max_iters=100, threshold=0.01):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param pred_col: Output prediction name (default,
         *'prediction_LogReg'*);
        :param alpha: Learning rate parameter (default, 0.1);
        :param regularization: Regularization parameter (default, 0.1);
        :param max_iters: Maximum number of iterations (default, 100);
        :param threshold: Tolerance for stopping criterion (default, 0.01);
        """
        super(LogisticRegression, self).__init__()

        if not pred_col:
            pred_col = 'prediction_LogReg'

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['pred_col'] = pred_col
        self.settings['alpha'] = alpha
        self.settings['regularization'] = regularization
        self.settings['threshold'] = threshold
        self.settings['iters'] = max_iters

        self.model = []
        self.name = 'LogisticRegression'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']
        alpha = self.settings.get('alpha', 0.1)
        reg = self.settings.get('regularization', 0.1)
        iters = self.settings.get('iters', 100)
        threshold = self.settings.get('threshold', 0.001)

        parameters = _logr_compute_coeffs(df, col_feature, col_label, alpha,
                                          iters, threshold, reg, nfrag)

        self.model = [compss_wait_on(parameters)]
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

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            result[i], info[i] = _logr_predict(df[i], self.settings,
                                               self.model[0])

        uuid_key = self._ddf_add_task(task_name='task_transform_logr',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


def _logr_sigmoid(x, w):
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


def _logr_compute_coeffs(data, features, label, alpha, iters,
                         threshold, reg, nfrag):
    """
    Perform a logistic regression via gradient ascent.
    """

    theta = np.array(np.zeros(1), dtype=float)  # initial
    i = reg = 0
    converged = False
    while (i < iters) and not converged:
        # grEin = gradient of in-sample Error
        gr_in = [_logr_gradient_ascent(data[f], features, label, theta, alpha)
                 for f in range(nfrag)]
        grad = merge_reduce(_logr_agg_sga, gr_in)
        result = _logr_calc_theta(grad, alpha, i, reg, threshold)
        i += 1
        theta, converged = result

    return theta


@task(returns=1)
def _logr_gradient_ascent(data, X, Y, theta, alfa):
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
    ys = data[Y].values
    for n in range(size):
        xn = np.array(xs[n, :])
        grad_p = _logr_sigmoid(xn, theta)
        gradient += xn*(ys[n] - grad_p)

    return [gradient, size, dim, theta]


@task(returns=list)
def _logr_agg_sga(info1, info2):

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


@local
def _logr_calc_theta(info, coef_lr, it, regularization, threshold):
    gradient = info[0]
    theta = info[3]

    # update coefficients
    alpha = coef_lr/(1+it)
    theta += alpha*(gradient - regularization*theta)

    converged = False
    if alpha*sum(gradient*gradient) < threshold:
        converged = True
    return [theta, converged]


@task(returns=2)
def _logr_predict(data, settings, theta):

    col_features = settings['feature_col']
    pred_col = settings['pred_col']
    size = len(data)

    xs = np.c_[np.ones(size), np.array(data[col_features].tolist())]
    data[pred_col] = [int(round(_logr_sigmoid(x, theta))) for x in xs]

    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
    return data, info


class GaussianNB(ModelDDF):
    """
    The Naive Bayes algorithm is an intuitive method that uses the
    probabilities of each attribute belonged to each class to make a prediction.
    It is a supervised learning approach that you would  come up with if you
    wanted to model a predictive probabilistically modeling problem.

    Naive bayes simplifies the calculation of probabilities by assuming that
    the probability of each attribute belonging to a given class value is
    independent of all other attributes. The probability of a class value given
    a value of an attribute is called the conditional probability. By
    multiplying the conditional probabilities together for each attribute for
    a given class value, we have the probability of a data instance belonging
    to that class.

    To make a prediction we can calculate probabilities of the instance
    belonged to each class and select the class value with the highest
    probability.

    :Example:

    >>> nb = GaussianNB(feature_col='features', label_col='label').fit(ddf1)
    >>> ddf2 = nb.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, pred_col=None):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param pred_col: Output prediction name
         (default, *'prediction_GaussianNB'*);
        """
        super(GaussianNB, self).__init__()

        if not pred_col:
            pred_col = 'prediction_GaussianNB'

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['pred_col'] = pred_col

        self.model = []
        self.name = 'GaussianNB'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        cols = [self.settings['feature_col'], self.settings['label_col']]

        # generate a dictionary with the sum of all attributes separed by class
        clusters, means = [[] for _ in range(nfrag)], [[] for _ in range(nfrag)]
        for i in range(nfrag):
            clusters[i], means[i] = _nb_separate_by_class(df[i], cols)

        # generate a total sum and len of each class
        mean = merge_reduce(_nb_merge_means, means)

        # compute the partial variance
        partial_var = [_nb_calc_var(mean, clusters[i]) for i in range(nfrag)]
        merged_fitted = merge_reduce(_nb_merge_var, partial_var)

        # compute standard deviation
        summaries = _nb_calc_stdev(merged_fitted)

        self.model = {'model': summaries}

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

        df, nfrag, tmp = self._ddf_inital_setup(data)

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            result[i], info[i] = _nb_predict_chunck(df[i], self.model['model'],
                                                    self.settings)

        uuid_key = self._ddf_add_task(task_name='task_transform_nb',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=2)
def _nb_separate_by_class(df_train, cols):
    """Sumarize all label in each fragment."""
    features, label = cols
    # a dictionary where:
    #   - keys are unique labels;
    #   - values are list of features
    summaries1 = {}
    means = {}
    for key in df_train[label].unique():
        values = df_train.loc[df_train[label] == key][features].tolist()
        summaries1[key] = values
        means[key] = [len(values), np.sum(values, axis=0)]

    return summaries1, means


@task(returns=1)
def _nb_merge_means(summ1, summ2):
    """Merge the statitiscs about each class (without variance)."""

    for att in summ2:
        if att in summ1:
            summ1[att] = [summ1[att][0] + summ2[att][0],
                          np.add(summ1[att][1], summ2[att][1])]
        else:
            summ1[att] = summ2[att]

    return summ1


@task(returns=1)
def _nb_calc_var(mean, separated):
    """Calculate the variance of each class."""

    summary = {}
    for key in separated:
        size, summ = mean[key]
        avg_class = np.divide(summ, size)
        errors = np.subtract(separated[key], avg_class)
        sse_partial = np.sum(np.power(errors, 2), axis=0)
        summary[key] = [sse_partial, size, avg_class]

    return summary


@task(returns=1)
def _nb_merge_var(summ1, summ2):
    """Merge the statitiscs about each class (with variance)."""
    for att in summ2:
        if att in summ1:
            summ1[att] = [np.add(summ1[att][0], summ2[att][0]),
                          summ1[att][1], summ1[att][2]]
        else:
            summ1[att] = summ2[att]
    return summ1


@local
def _nb_calc_stdev(summaries):
    """Calculate the standart desviation of each class."""
    new_summaries = {}
    for att in summaries:
        sse_partial, size, avg = summaries[att]
        std = np.sqrt(np.divide(sse_partial, size))
        new_summaries[att] = [avg, std]

    return new_summaries


@task(returns=2)
def _nb_predict_chunck(data, summaries, settings):
    """Predict all records in a fragment."""

    features_col = settings['feature_col']
    predicted_label = settings['pred_col']
    pi = 3.1415926535
    n = len(data)
    data.reset_index(drop=True, inplace=True)

    for class_v, class_summaries in summaries.iteritems():
        avgs, stds = class_summaries
        dim = len(avgs)
        dens, nums = np.zeros(dim), np.zeros(dim)

        for i, (avg, std) in enumerate(zip(avgs, stds)):
            if std == 0:
                std = 0.0000001

            dens[i] = (2 * pow(std, 2))
            nums[i] = np.divide(1.0, (math.sqrt(2*pi) * std))

        summaries[class_v] = [avgs, dens, nums]

    predictions = np.zeros(n, dtype=int)
    for i in range(n):
        predictions[i] = _nb_predict(summaries, data[features_col].iat[i])

    data[predicted_label] = predictions.tolist()

    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
    return data, info


def _nb_predict(summaries, input_vector):
    """Predict a feature."""
    probabilities = _nb_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for classValue, probability in probabilities.iteritems():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = classValue
    return best_label


def _nb_class_probabilities(summaries, x):
    """Do the probability's calculation of all records."""
    probs = {}

    for class_v, class_summaries in summaries.iteritems():
        probs[class_v] = 1
        avgs, dens, nums = class_summaries
        for i, (avg,  den, num) in enumerate(zip(avgs, dens, nums)):
            probs[class_v] *= num * math.exp(-(pow(x[i] - avg, 2) / den))

    return probs

