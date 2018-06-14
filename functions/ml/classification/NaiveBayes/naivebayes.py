#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Gaussian Naive Bayes.

The Naive Bayes algorithm is an intuitive method that uses the probabilities
of each attribute belonged to each class to make a prediction. It is a
supervised learning approach that you would  come up with if you wanted to
model a predictive probabilistically modeling problem.

Naive bayes simplifies the calculation of probabilities by assuming that the
probability of each attribute belonging to a given class value is independent
of all other attributes. The probability of a class value given a value of an
attribute is called the conditional probability. By multiplying the conditional
probabilities together for each attribute for a given class value, we have the
probability of a data instance belonging to that class. To make a prediction we
can calculate probabilities of the instance belonged to each class and select
the class value with the highest probability.
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import math
import pandas as pd
from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce


class GaussianNB(object):
    """Gaussian Naive Bayes's methods.

    - fit()
    - transform()
    """

    def fit(self, data, settings, nfrag):
        """Fit.

        :param data:      A list with nfrag pandas's dataframe used to
                            training the model.
        :param settings:  A dictionary that contains:
           - features: 	    Field of the features in the training data;
           - label:         Field of the labels   in the training data;
        :param nfrag:   A number of fragments;
        :return:          The model created (which is a pandas dataframe).
        """
        if 'features' not in settings or 'label' not in settings:
            raise Exception("You must inform the `features` "
                            "and `label` fields.")

        label = settings['label']
        features = settings['features']

        # first analysis
        separated = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            separated[i] = _separateByClass(data[i], label, features)

        # generate a mean and len of each fragment
        merged_fitted = mergeReduce(_merge_summaries1, separated)

        # compute the variance
        partial_result = [_addVar(merged_fitted, separated[i])
                          for i in range(nfrag)]
        merged_fitted = mergeReduce(_merge_summaries2, partial_result)
        summaries = _calcSTDEV(merged_fitted)

        from pycompss.api.api import compss_wait_on
        summaries = compss_wait_on(summaries)

        model = dict()
        model['algorithm'] = 'GaussianNB'
        model['model'] = summaries
        return model

    def transform(self, data, model, settings, nfrag):
        """Transform.

        :param data:    A list with nfrag pandas's dataframe that
                        will be predicted.
        :param model: A model already trained;
        :param settings: A dictionary that contains:
            - features: Field of the features in the test data;
            - predCol: Alias to the new column with the labels predicted;
        :param nfrag: A number of fragments;
        :return: The prediction (in the same input format).
        """
        if 'features' not in settings:
            raise Exception("You must inform the at "
                            "least the `features` field.")

        if model.get('algorithm', 'null') != 'GaussianNB':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _predict_chunck(data[f], model, settings)

        return result

    def transform_serial(self, data, model, settings):
        """Transform.

        :param data: A DataFrame that will be predicted.
        :param model: A model already trained;
        :param settings: A dictionary that contains:
            - features: Field of the features in the test data;
            - predCol: Alias to the new column with the labels predicted;
        :return: The prediction (in the same input format).
        """
        if 'features' not in settings:
            raise Exception("You must inform the at "
                            "least the `features` field.")

        if model.get('algorithm', 'null') != 'GaussianNB':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = _predict_chunck_(data, model, settings)

        return result


@task(returns=list)
def _separateByClass(train_data, label, features):
    """Sumarize all label in each fragment."""
    separated = {}
    for i in range(len(train_data)):
        element = train_data.iloc[i][label]
        if element not in separated:
            separated[element] = []
        separated[element].append(train_data.iloc[i][features])

    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = _summarize(instances)

    return summaries


@task(returns=dict)
def _addVar(merged_fitted, separated):
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
def _calcSTDEV(summaries):
    """Calculate the standart desviation of each class."""
    new_summaries = {}
    for att in summaries:
        tupla = summaries[att]
        new_summaries[att] = []
        for t in tupla:
            new_summaries[att].append((t[0], math.sqrt(t[1]/t[2])))
    return new_summaries


@task(returns=list)
def _merge_summaries2(summ1, summ2):
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
def _merge_summaries1(summ1, summ2):
    """Merge the statitiscs about each class (without variance)."""
    for att in summ2:
        if att in summ1:
            for i in range(len(summ1[att])):
                summ1[att][i] = (summ1[att][i][0] + summ2[att][i][0],
                                 summ1[att][i][1] + summ2[att][i][1])
        else:
            summ1[att] = summ2[att]

    return summ1


def _summarize(features):
    """Append the sum and the length of each class."""
    summaries = []
    for attribute in zip(*features):
        avg = sum(attribute)
        summaries.append((avg, len(attribute)))
    return summaries


@task(returns=list)
def _predict_chunck(data, summaries, settings):
    """Predict all records in a fragment."""
    return _predict_chunck_(data, summaries, settings)


def _predict_chunck_(data, summaries, settings):
    """Predict all records in a fragment."""
    features_col = settings['features']
    predicted_label = settings.get('predCol', 'prediction')

    predictions = []
    for i in range(len(data)):
        result = _predict(summaries, data.iloc[i][features_col])
        predictions.append(result)

    data[predicted_label] = pd.Series(predictions).values
    return data


def _predict(summaries, inputVector):
    """Predict a feature."""
    probabilities = _calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def _calculateClassProbabilities(summaries, toPredict):
    """Do the probability's calculation of all records."""
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, var = classSummaries[i]
            x = toPredict[i]
            probabilities[classValue] *= _calculateProbability(x, mean, var)
    return probabilities


def _calculateProbability(x, mean, stdev):
    """Do the probability's calculation of one record."""
    # exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    # prob = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    import functions_naivebayes
    prob = functions_naivebayes.calculateProbability(x, mean, stdev)
    return prob
