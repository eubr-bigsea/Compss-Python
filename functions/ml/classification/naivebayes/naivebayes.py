#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


import random
import math
import numpy as np
import pandas as pd
import time

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.functions.data    import chunks
from pycompss.api.api import compss_wait_on






# #============ To Profile ===================
# def timing(f):
#     def wrap(*args):
#         time1 = time.time()
#         ret = f(*args)
#         time2 = time.time()
#         print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
#         return ret
#     return wrap
# #=========================================

#-------------------------------------------------------------------------
#   Naive Bayes
#
#   Create the model
#-------------------------------------------------------------------------

class GaussianNB(object):

    def fit(self,data,settings,numFrag):
        """
            Gaussian Naive Bayes:

            The Naive Bayes algorithm is an intuitive method
            that uses the probabilities of each attribute belonging to each class
            to make a prediction. It is the supervised learning approach you would
            come up with if you wanted to model a predictive modeling problem
            probabilistically.

            Naive bayes simplifies the calculation of probabilities by assuming that
            the probability of each attribute belonging to a given class value is
            independent of all other attributes.

            The probability of a class value given a value of an attribute is called
            the conditional probability. By multiplying the conditional
            probabilities together for each attribute for a given class value, we
            have a probability of a data instance belonging to that class.

            :param train_data: A np.array (splitted)
            :param numFrag: num fragments, if -1 data is considered chunked
            :return The model (np.array)
        """

		#Data format:  label,f1,f2,f3...
        columns = settings['labels']+settings['features']


        train_data		= [ self.format_data(data[i],columns) for i in range(numFrag)]
        separated       = [ self.separateByClass(train_data[i]) for i in range(numFrag)]   # separa as classes
        partial_fitted  = [ self.partial_fit(separated[i]) for i in range(numFrag)]
        merged_fitted   = mergeReduce(self.merge_summaries1, partial_fitted) #result: mean and len

        partial_result  = [ self.addVar(merged_fitted,separated[i])  for i in range(numFrag)]
        merged_fitted   = mergeReduce(self.merge_summaries2, partial_result)

        merged_fitted = compss_wait_on(merged_fitted)
        summaries = self.calcSTDEV(merged_fitted)

        #self.summaries = summaries
        return summaries

    @task(returns=list, isModifier = False)
    def format_data(self,data,columns):
        return  data[columns].values


    @task(returns=dict, isModifier = False)
    def addVar(self,merged_fitted, separated):

        summary = {}
        for att in separated:
            summaries = []
            nums = separated[att]
            print "nums: ",nums
            d = 0
            nums2 = merged_fitted[att]
            #print "nums2: ",nums2
            for attribute in  zip(*nums):
                avg = nums2[d][0]/nums2[d][1]
                #print avg
                varNum = sum([math.pow(x-avg,2) for x in attribute])
                summaries.append((avg, varNum, nums2[d][1]))
                d+=1
            summary[att] = summaries

        return summary


    def calcSTDEV(self,summaries):
        #summaries = summaries[0]
        new_summaries = {}
        for att in summaries:
            tupla = summaries[att]
            new_summaries[att]=[]
            for t in tupla:
                new_summaries[att].append((t[0], math.sqrt(t[1]/t[2])))
        return new_summaries

    @task(returns=dict, isModifier = False)
    def partial_fit(self,separated):
        summaries = {}
        for classValue, instances in separated.iteritems():
            summaries[classValue] = self.summarize(instances)

        return summaries

    @task(returns=list, isModifier = False)
    def merge_summaries2(self,summaries1,summaries2):
        for att in summaries2:
            if att in summaries1:
                for i in range(len(summaries1[att])):
                    summaries1[att][i] = (summaries1[att][i][0],
                                          summaries1[att][i][1] + summaries2[att][i][1],
                                          summaries1[att][i][2] )

            else:
                summaries1[att] = summaries2[att]
        return summaries1

    @task(returns=list, isModifier = False)
    def merge_summaries1(self,summaries1, summaries2):
        for att in summaries2:
            if att in summaries1:
                for i in range(len(summaries1[att])):
                    summaries1[att][i] = ( (summaries1[att][i][0] + summaries2[att][i][0]),
                                           summaries1[att][i][1] + summaries2[att][i][1] )
            else:
                summaries1[att] = summaries2[att]

        return summaries1



    @task(returns=list, isModifier = False)
    def separateByClass(self,train_data):
    	separated = {}

    	for i in range(len(train_data)):
    	 	if (train_data[i][0] not in separated):
    	 		separated[train_data[i][0]] = []
    	 	separated[train_data[i][0]].append(train_data[i][1:,])

    	return separated

    def summarize(self,features):
        summaries = []
        for attribute in zip(*features):
            avg = sum(attribute)
            #varNum = sum([pow(x-avg,2) for x in attribute])
            summaries.append((avg, len(attribute)))
        return summaries




    #-------------------------------------------------------------------------
    #   Naive Bayes
    #
    #   predictions
    #-------------------------------------------------------------------------

    def transform(self,data,settings, numFrag):
        """
            Gaussian Naive Bayes:

            To make a prediction we can calculate probabilities of the instance
            belonging to each class and select the class value with the highest
            probability.

            :param TestSet:  A np.array (splitted) with the data
            :param model: A summaries, a np.array with the probabilities.
            :param numFrag: num fragments, if -1 data is considered chunked
            :return: list with the predictions.
        """



        partialResult = [ self.predict_chunck(settings, data[i])  for i in range(numFrag) ]
        #result = mergeReduce(self.merge_lists, partialResult)
        #result = compss_wait_on(result)

        return partialResult

    @task(returns=list, isModifier = False)
    def merge_lists(self,list1,list2):
        return list1+list2

    @task(returns=list, isModifier = False)
    def predict_chunck(self,settings, data):
        summaries =  settings['model']
        #print summaries
        features = settings['features']
        label = "_".join(i for i in settings['labels'])
        testSet = np.array(data[features].values)

        predictions = []
        for i in range(len(testSet)):
         	result = self.predict(summaries, testSet[i])
         	predictions.append(result)


		new_column = label +"_predited"
        data[new_column] =  pd.Series(predictions).values
        return data

    def predict(self,summaries, inputVector):
    	probabilities = self.calculateClassProbabilities(summaries, inputVector)
    	bestLabel, bestProb = None, -1
    	for classValue, probability in probabilities.iteritems():
    		if bestLabel is None or probability > bestProb:
    			bestProb = probability
    			bestLabel = classValue
    	return bestLabel


    def calculateClassProbabilities(self,summaries, toPredict):

        probabilities = {}
        for classValue, classSummaries in summaries.iteritems():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, var = classSummaries[i]
                x = toPredict[i]
                probabilities[classValue] *= self.calculateProbability(x, mean, var)
        return probabilities


    def calculateProbability(self,x, mean, stdev):
        #exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        #prob = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
        import functions_naivebayes
        prob = functions_naivebayes.calculateProbability(x,mean,stdev)
        return prob



def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
