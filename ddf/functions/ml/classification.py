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
from ddf.ddf import COMPSsContext, DDF, ModelDDF

__all__ = ['KNearestNeighbors', 'GaussianNB', 'LogisticRegression', 'SVM']


import uuid
import sys
sys.path.append('../../')


class KNearestNeighbors(ModelDDF):

    """K-Nearest Neighbor is a algorithm used that can be used for both
    classification and regression predictive problems. However, it is more
    widely used in classification problems. Is a non parametric lazy learning
    algorithm. Meaning that it does not use the training data points to do
    any generalization. In other words, there is no explicit training phase.
    More precisely, all the training data is needed during the testing phase.

    In a classification, the algorithm computes from a simple majority vote
    of the K nearest neighbors of each point present in the training set.
    The choice of the parameter K is very crucial in this algorithm, and
    depends on the dataset. However, values of one or tree is more commom.

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

        if not feature_col:
            raise Exception("You must inform the `features` field.")

        if not label_col:
            raise Exception("You must inform the `label` field.")

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

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']

        train_data = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            train_data[f] = _knn_create_model(df[f], col_label, col_feature)
        model = merge_reduce(merge_lists, train_data)

        self.model = [compss_wait_on(model)]
        return self

    def transform(self, data):
        """

        :param data: DDF
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        self.settings['model'] = self.model[0]

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        result = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            result[i] = _knn_classify_block_(df[i], self.settings)

        uuid_key = tmp._generate_uuid()
        COMPSsContext.tasks_map[uuid_key] = \
            {'name': 'task_transform_knn',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: result},
             'parent': [tmp.last_uuid],
             'output': 1,
             'input': 1
             }

        tmp._set_n_input(uuid_key, tmp.settings['input'])
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=list)
def _knn_create_model(df, label, features):
    """Create a partial model based in the selected columns."""
    labels = df[label].values
    feature = np.array(df[features].values.tolist())
    return [labels, feature]


@task(returns=list)
def merge_lists(list1, list2):
    """Merge all elements in an unique dataframe to be part of a knn model."""
    l1, f1 = list1
    l2, f2 = list2

    if len(l1) != 0:
        if len(l2) != 0:
            result1 = np.concatenate((l1, l2), axis=0)
            result2 = np.concatenate((f1, f2), axis=0)
            return [result1, result2]
        else:
            return list1
    else:
        return list2


@task(returns=1)
def _knn_classify_block_(data, settings):
    """Perform a partial classification."""
    col_features = settings['feature_col']
    pred_col = settings.get('pred_col', "prediction_kNN")
    K = settings.get('k', 3)
    model = settings['model']

    sizeTest = len(data)
    if sizeTest == 0:
        data[pred_col] = np.nan
        return data

    # initalizing variables
    sample = data[col_features].values[0]
    if isinstance(sample, list):
        numDim = len(sample)
    else:
        numDim = 1

    semi_labels = [[0 for _ in range(K)] for _ in range(sizeTest)]

    # from ddf.functions.ml import functions_knn
    from timeit import default_timer as timer
    start = timer()

    semi_labels = dist2all(
                  model[1], np.array(data.iloc[:][col_features].tolist()),
                  numDim, K, semi_labels, model[0])
    end = timer()
    print "{0:.2e}".format(end - start)
    values = _knn_get_majority(semi_labels)
    data[pred_col] = pd.Series(values).values

    return data


def dist2all(dataTrain, dataTest, numDim,  K, semi_labels, model):

    sizeTest = dataTest.shape[0]
    sizeTrain = dataTrain.shape[0]

    for i_test in range(sizeTest):
        semi_dist = np.empty(sizeTrain)
        for i_train in range(sizeTrain):
            semi_dist[i_train] = \
                np.linalg.norm(dataTrain[i_train]-dataTest[i_test])
        idx = np.argpartition(semi_dist, K)
        semi_labels[i_test] = model[idx[:K]]

    return semi_labels


def _knn_get_majority(neighborhood):
    """Finding the most frequent label."""
    result = np.zeros(len(neighborhood)).tolist()
    for i in range(len(neighborhood)):
        labels = neighborhood[i].tolist()
        result[i] = \
            max(map(lambda val: (labels.count(val), val), set(labels)))[1]
    return result


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

    The algorithm reads a dataset composed by labels (-1.0 or 1.0) and
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

        if not feature_col:
            raise Exception("You must inform the `features` field.")

        if not label_col:
            raise Exception("You must inform the `label` field.")

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

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']

        w = [0]
        old_cost = np.inf

        for it in range(coef_max_iter):
            cost_grad_p = [_calc_cost_grad(df[f], f, coef_lambda,
                                           w, col_label, col_feature) for f in
                           range(nfrag)]
            cost_grad = merge_reduce(_accumulate_cost_grad, cost_grad_p)
            cost_grad = compss_wait_on(cost_grad)

            cost = cost_grad[0]
            thresold = np.abs(old_cost - cost)
            if thresold <= coef_threshold:
                print "[INFO] - Final Cost %.4f" % (cost)
                break
            else:
                old_cost = cost

            w = _update_weight(coef_lr, cost_grad, w)

        self.model = [w]

        return self

    def transform(self, data):
        """

        :param data: DDF
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        predicted_label = self.settings['pred_col']
        features = self.settings['feature_col']

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _predict_partial(df[f], self.model[0],
                                         predicted_label, features)

        uuid_key = tmp._generate_uuid()
        COMPSsContext.tasks_map[uuid_key] = \
            {'name': 'task_transform_svm',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: result},
             'parent': [tmp.last_uuid],
             'output': 1,
             'input': 1
             }

        tmp._set_n_input(uuid_key, tmp.settings['input'])
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@local
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
                if (condition - 1) < 0:
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

    def _predict_one(test_xi, w):
        """Predict one record based in the model."""
        if np.matmul(test_xi, w) >= 0:
            return 1.0
        return -1.0

    if len(data) > 0:
        values = \
            np.vectorize(_predict_one, excluded=['w'])(test_xi=data[features],
                                                       w=w)
        data[predicted_label] = values.tolist()
    else:
        data[predicted_label] = np.nan

    return data


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

        if not feature_col:
            raise Exception("You must inform the `features` field.")

        if not label_col:
            raise Exception("You must inform the `label` field.")

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

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

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

    def transform(self, data):
        """

        :param data: DDF
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        result = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            result[i] = _logr_predict(df[i], self.settings, self.model[0])

        uuid_key = tmp._generate_uuid()
        COMPSsContext.tasks_map[uuid_key] = \
            {'name': 'task_transform_logr',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: result},
             'parent': [tmp.last_uuid],
             'output': 1,
             'input': 1
             }

        tmp._set_n_input(uuid_key, tmp.settings['input'])
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


@task(returns=list)
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

    for n in range(size):
        xn = np.array(xs[n, :])
        yn = data[Y].values[n]
        grad_p = _logr_sigmoid(xn, theta)
        gradient += xn*(yn - grad_p)

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


@task(returns=list)
def _logr_predict(data, settings, theta):

    col_features = settings['feature_col']
    pred_col = settings['pred_col']
    size = len(data)

    xs = np.c_[np.ones(size), np.array(data[col_features].tolist())]
    data[pred_col] = [round(_logr_sigmoid(x, theta)) for x in xs]
    return data


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

        if not feature_col:
            raise Exception("You must inform the `features` field.")

        if not label_col:
            raise Exception("You must inform the `label` field.")

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

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']
        # first analysis
        separated = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            separated[i] = _nb_separateByClass(df[i], col_label, col_feature)

        # generate a mean and len of each fragment
        merged_fitted = merge_reduce(_nb_merge_summaries1, separated)

        # compute the variance
        partial_result = [_nb_addVar(merged_fitted, separated[i])
                          for i in range(nfrag)]
        merged_fitted = merge_reduce(_nb_merge_summaries2, partial_result)
        summaries = _nb_calcSTDEV(merged_fitted)

        self.model = [compss_wait_on(summaries)]
        return self

    def transform(self, data):
        """
        :param data: DDF
        :return: DDF
        """
        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        result = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            result[i] = _nb_predict_chunck(df[i], self.model[0], self.settings)

        uuid_key = tmp._generate_uuid()
        COMPSsContext.tasks_map[uuid_key] = \
            {'name': 'task_transform_nb',
             'status': 'COMPLETED',
             'lazy': False,
             'function': {0: result},
             'parent': [tmp.last_uuid],
             'output': 1,
             'input': 1
             }

        tmp._set_n_input(uuid_key, tmp.settings['input'])
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=list)
def _nb_separateByClass(train_data, label, features):
    """Sumarize all label in each fragment."""
    separated = {}
    for i in range(len(train_data)):
        element = train_data.iloc[i][label]
        if element not in separated:
            separated[element] = []
        separated[element].append(train_data.iloc[i][features])

    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = _nb_summarize(instances)

    return summaries


@task(returns=dict)
def _nb_addVar(merged_fitted, separated):
    """Calculate the variance of each class."""
    summary = {}
    for att in separated:
        summaries = []
        nums = separated[att]
        d = 0
        nums2 = merged_fitted[att]
        for attribute in zip(*nums):
            avg = nums2[d][0]/nums2[d][1]
            var_num = sum([math.pow(x-avg, 2) for x in attribute])
            summaries.append((avg, var_num, nums2[d][1]))
            d += 1
        summary[att] = summaries

    return summary


@task(returns=dict)
def _nb_calcSTDEV(summaries):
    """Calculate the standart desviation of each class."""
    new_summaries = {}
    for att in summaries:
        tupla = summaries[att]
        new_summaries[att] = []
        for t in tupla:
            new_summaries[att].append((t[0], math.sqrt(t[1]/t[2])))
    return new_summaries


@task(returns=list)
def _nb_merge_summaries2(summ1, summ2):
    """Merge the statitiscs about each class (with variance)."""
    for att in summ2:
        if att in summ1:
            for i in range(len(summ1[att])):
                tmp = summ1[att][i][1] + summ2[att][i][1]
                summ1[att][i] = (summ1[att][i][0], tmp, summ1[att][i][2])
        else:
            summ1[att] = summ2[att]
    return summ1


@task(returns=list)
def _nb_merge_summaries1(summ1, summ2):
    """Merge the statitiscs about each class (without variance)."""
    for att in summ2:
        if att in summ1:
            for i in range(len(summ1[att])):
                summ1[att][i] = (summ1[att][i][0] + summ2[att][i][0],
                                 summ1[att][i][1] + summ2[att][i][1])
        else:
            summ1[att] = summ2[att]

    return summ1


def _nb_summarize(features):
    """Append the sum and the length of each class."""
    summaries = []
    for attribute in zip(*features):
        avg = sum(attribute)
        summaries.append((avg, len(attribute)))
    return summaries


@task(returns=1)
def _nb_predict_chunck(data, summaries, settings):
    """Predict all records in a fragment."""

    features_col = settings['feature_col']
    predicted_label = settings['pred_col']

    predictions = []
    for i in range(len(data)):
        result = _nb_predict(summaries, data.iloc[i][features_col])
        predictions.append(result)

    data[predicted_label] = predictions

    return data


def _nb_predict(summaries, inputVector):
    """Predict a feature."""
    probabilities = _nb_calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def _nb_calculateClassProbabilities(summaries, toPredict):
    """Do the probability's calculation of all records."""
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, var = classSummaries[i]
            x = toPredict[i]
            probabilities[classValue] *= _nb_calculateProbability(x, mean, var)
    return probabilities


def _nb_calculateProbability(x, mean, stdev):
    """Do the probability's calculation of one record."""
    # exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    # prob = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    # import functions_naivebayes
    prob = calculateProbability(x, mean, stdev)
    return prob


def calculateProbability(x, mean, stdev):
    pi = 3.14159265358979323846

    result = 0
    exponent = math.exp(-(pow(x-mean, 2)/(2*pow(stdev, 2))))

    if stdev == 0:
        stdev = 0.0000001

    result = (1.0 / (math.sqrt(2*pi) * stdev)) * exponent
    return result
