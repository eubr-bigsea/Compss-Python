#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import random
import math
import numpy as np


#-------------------------------------------------------------------------




#-------------------------------------------------------------------------

def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        row =  []
        label = float(dataset[i][0])
        f = [float(x) for x in dataset[i][1:len(dataset[i])]]
        dataset[i] = [label,f]
    return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


#
#   Naive Bayes
#

def separateByClass(features,labels):
	separated = {}

	for i in range(len(labels)):
	 	if (labels[i] not in separated):
	 		separated[labels[i]] = []
	 	separated[labels[i]].append(features[i])

	return separated

def summarize(features):
	summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*features)]
	#del summaries[0] #remove the class
	return summaries

def summarizeByClass(features,labels):
	separated = separateByClass(features,labels)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries


#usar Cython
def calculateProbability(x, mean, stdev):
    if  stdev == 0: stdev = 0.000001

    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, toPredict):

    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, var = classSummaries[i]
            x = toPredict[i]
            probabilities[classValue] *= calculateProbability(x, mean, var)
    return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	#print (probabilities)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0



def main():
    filename = 'teste.csv'
    splitRatio = 0.67
    trainingSet = loadCsv("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/Iris/iris_Train.data")
    testSet     = loadCsv("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/Iris/iris_test.data")

    #trainingSet = loadCsv("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/higgs/train_0.012m.csv")
    #testSet     = loadCsv("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/higgs/train_0.012m.csv")
    #dataset  = loadCsv(filename)
    #trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Train={0} and Test={1} rows').format(len(trainingSet), len(testSet))
	# prepare model

    x = [i[1] for i in trainingSet]
    y = [i[0] for i in trainingSet]

    summaries = summarizeByClass(x,y)
    print summaries
	# test model
    test_x = [i[1] for i in testSet]
    test_y = [i[0] for i in testSet]
    predictions = getPredictions(summaries, test_x)
    #
    print "predictions"
    accuracy = getAccuracy(test_y, predictions)
    print('Accuracy: {0}%').format(accuracy)

	# from sklearn.naive_bayes import GaussianNB
    #
	# X,y = SplitXy(trainingSet)
    #
	# X = X.reshape(-1, 1)
    #
	# model = GaussianNB()
	# model.fit(X, y)
    #
    # ### Compare the models built by Python
    #
	# print ("Class: -1")
	# for i,j in enumerate(model.theta_[0]):
	# 	print ("({:8.2f} {:9.2f} {:7.2f} )".format(j, model.sigma_[0][i], math.sqrt(model.sigma_[0][i])) , "")
	# #	print ("==> ", summaries[0][i])
    #
	# print ("Class: 1")
	# for i,j in enumerate(model.theta_[1]):
	# 	print ("({:8.2f} {:9.2f} {:7.2f} )".format(j, model.sigma_[1][i], math.sqrt(model.sigma_[1][i])) , "")
	# #	print ("==> ", summaries[1][i])

main()
