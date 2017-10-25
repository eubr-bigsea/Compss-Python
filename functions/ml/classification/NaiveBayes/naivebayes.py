#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

import math
import numpy as np
import pandas as pd

from pycompss.api.parameter     import *
from pycompss.api.task          import task
from pycompss.functions.reduce  import mergeReduce

#-------------------------------------------------------------------------
#   Naive Bayes
#
#-------------------------------------------------------------------------

class GaussianNB(object):

    """
    Gaussian Naive Bayes:

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

        Methods:
            - fit()
            - transform()


    """

    def fit(self, data, settings, numFrag):
        """
            fit():
            - :param data:      A list with numFrag pandas's dataframe used to
                                training the model.
            - :param settings:  A dictionary that contains:
             	- features: 	Field of the features in the training data;
             	- label:        Field of the labels   in the training data;
            - :param numFrag:   A number of fragments;
            - :return:          The model created (which is a pandas dataframe).
        """

        if 'features' not in settings or  'label'  not in settings:
           raise Exception("You must inform the `features` and `label` fields.")

        label = settings['label']
        features = settings['features']

        #first analysis
        separated  = [[] for i in range(numFrag)]
        for i in range(numFrag):
            separated[i] = separateByClass(data[i], label, features)

        #generate a mean and len of each fragment
        merged_fitted = mergeReduce(merge_summaries1, separated )

        #compute the variance
        partial_result = [ addVar(merged_fitted,separated[i])
                            for i in range(numFrag)]
        merged_fitted  = mergeReduce(merge_summaries2, partial_result)

        summaries = calcSTDEV(merged_fitted)

        model = {}
        model['algorithm'] = 'GaussianNB'
        model['model'] = summaries
        return model

    def transform(self, data, model, settings, numFrag):
        """
            transform:
            - :param data:  A list with numFrag pandas's dataframe that
                            will be predicted.
            - :param model: A model already trained;
            - :param settings: A dictionary that contains:
             	- features: Field of the features in the test data;
             	- predCol: Alias to the new column with the labels predicted;
            - :param numFrag: A number of fragments;
            - :return: The prediction (in the same input format).
        """

        if 'features' not in settings:
           raise Exception("You must inform the at least the `features` field.")

        if model.get('algorithm','null') != 'GaussianNB':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = [[] for i in range(numFrag)]
        for f in range(numFrag):
            result[f] = predict_chunck(data[f], model, settings)

        return result

@task(returns=list)
def separateByClass(train_data,label,features):
    separated = {}
    for i in range(len(train_data)):
        l = train_data.iloc[i][label]
        if (l not in separated):
            separated[l] = []
        separated[l].append(train_data.iloc[i][features])

    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)

    return summaries


@task(returns=dict)
def addVar(merged_fitted, separated):

    summary = {}
    for att in separated:
        summaries = []
        nums = separated[att]
        d = 0
        nums2 = merged_fitted[att]
        for attribute in  zip(*nums):
            avg = nums2[d][0]/nums2[d][1]
            varNum = sum([math.pow(x-avg,2) for x in attribute])
            summaries.append((avg, varNum, nums2[d][1]))
            d+=1
        summary[att] = summaries

    return summary

@task(returns=dict)
def calcSTDEV(summaries):
    new_summaries = {}
    for att in summaries:
        tupla = summaries[att]
        new_summaries[att]=[]
        for t in tupla:
            new_summaries[att].append((t[0], math.sqrt(t[1]/t[2])))
    return new_summaries



@task(returns=list)
def merge_summaries2(summaries1,summaries2):
    for att in summaries2:
        if att in summaries1:
            for i in range(len(summaries1[att])):
                summaries1[att][i] = (summaries1[att][i][0],
                            summaries1[att][i][1] + summaries2[att][i][1],
                            summaries1[att][i][2] )

        else:
            summaries1[att] = summaries2[att]
    return summaries1

@task(returns=list)
def merge_summaries1(summaries1, summaries2):
    for att in summaries2:
        if att in summaries1:
            for i in range(len(summaries1[att])):
                summaries1[att][i] = (
                            (summaries1[att][i][0] + summaries2[att][i][0]),
                             summaries1[att][i][1] + summaries2[att][i][1])
        else:
            summaries1[att] = summaries2[att]

    return summaries1

def summarize(features):
    summaries = []
    for attribute in zip(*features):
        avg = sum(attribute)
        summaries.append((avg, len(attribute)))
    return summaries




#-------------------------------------------------------------------------
#   Naive Bayes
#
#   predictions
#-------------------------------------------------------------------------



@task(returns=list)
def merge_lists(list1,list2):
    return list1+list2

@task(returns=list)
def predict_chunck(data, summaries, settings):

    features_col    = settings['features']
    predictedLabel  = settings.get('predCol','prediction')

    predictions = []
    for i in range(len(data)):
     	result = predict(summaries, data.iloc[i][features_col])
     	predictions.append(result)


    data[predictedLabel] =  pd.Series(predictions).values
    return data

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


def calculateClassProbabilities(summaries, toPredict):

    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, var = classSummaries[i]
            x = toPredict[i]
            probabilities[classValue]  *=  calculateProbability(x, mean, var)
    return probabilities


def calculateProbability(x, mean, stdev):
    #exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    #prob = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    import functions_naivebayes
    prob = functions_naivebayes.calculateProbability(x,mean,stdev)
    return prob
