#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF, generate_info
from ddf_library.ddf_model import ModelDDF

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
# from pycompss.api.local import *

import math
import numpy as np
import pandas as pd


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

        if not isinstance(feature_col, list):
            feature_col = [feature_col]

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
        clusters = [[] for _ in range(nfrag)]
        means = [[] for _ in range(nfrag)]
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
        for f in range(nfrag):
            result[f], info[f] = _nb_predict_chunck(df[f], self.model['model'],
                                                    self.settings, f)

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
        values = df_train.loc[df_train[label] == key][features].values
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


# @local
def _nb_calc_stdev(summaries):
    """Calculate the standart desviation of each class."""
    summaries = compss_wait_on(summaries)
    new_summaries = {}
    for att in summaries:
        sse_partial, size, avg = summaries[att]
        # std = np.sqrt(np.divide(sse_partial, size))
        var = np.divide(sse_partial, size)
        new_summaries[att] = [avg, var, size]

    return new_summaries


@task(returns=2)
def _nb_predict_chunck(data, summaries, settings, frag):
    """Predict all records in a fragment."""

    features_col = settings['feature_col']
    predicted_label = settings['pred_col']
    # pi = 3.1415926

    data.reset_index(drop=True, inplace=True)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    # list of unique classes
    clf.classes_ = np.array([key for key in summaries])

    # probability of each class
    clf.class_prior_ = np.array([0.5 for _ in summaries])

    # mean of each feature per class
    clf.theta_ = np.array([summaries[i][0] for i in summaries])

    # variance of each feature per class
    clf.sigma_ = np.array([summaries[i][1] for i in summaries])

    array = data[features_col].values
    predictions = clf.predict(array)

    # for class_v, class_summaries in summaries.items():
    #     avgs, stds, _ = class_summaries
    #     dim = len(avgs)
    #     dens, nums = np.zeros(dim), np.zeros(dim)
    #
    #     for i, (avg, std) in enumerate(zip(avgs, stds)):
    #         if std == 0:
    #             std = 0.0000001
    #
    #         dens[i] = (2 * pow(std, 2))
    #         nums[i] = np.divide(1.0, (math.sqrt(2*pi) * std))
    #
    #     summaries[class_v] = [avgs, dens, nums]
    #
    # array = data[features_col].values
    #
    # def _nb_predict(input_vector, model=summaries):
    #     """Predict a feature."""
    #     probabilities = _nb_class_probabilities(model, input_vector)
    #     best_label, best_prob = None, -1
    #     for classValue in probabilities:
    #         probability = probabilities[classValue]
    #         if best_label is None or probability > best_prob:
    #             best_prob = probability
    #             best_label = classValue
    #     return best_label
    #
    # predictions = np.apply_along_axis(_nb_predict, 1, array)

    if predicted_label in data:
        data.drop([predicted_label], axis=1, inplace=True)

    data[predicted_label] = predictions  #.tolist()

    info = generate_info(data, frag)
    return data, info


# def _nb_class_probabilities(summaries, x):
#     """Do the probability's calculation of all records."""
#     probs = {}
#
#     for class_v, class_summaries in summaries.items():
#         probs[class_v] = 1
#         avgs, dens, nums = class_summaries
#         probs[class_v] = np.prod(nums * np.exp(-(pow(x - avgs, 2) / dens)))
#
#     return probs