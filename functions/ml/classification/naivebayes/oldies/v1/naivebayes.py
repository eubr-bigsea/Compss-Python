#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


import csv
import random
import math
import numpy as np
import time



#
#       Versao 1.0: A predicao está paralela, no entando para criar o modelo ainda está serial  ---> O Master teria q ler todos os dados
#
#



from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.functions.data    import chunks

#============ To Profile ===================
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap
#=========================================

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


#-------------------------------------------------------------------------
#   Naive Bayes
#
#   Create the model
#-------------------------------------------------------------------------


def fit(features,labels):
    separated = separateByClass(features,labels)  # separa as classes
    #print separated
    #print "------------"
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
        print summaries[classValue]
    return summaries



def separateByClass(features,labels):
	separated = {}

	for i in range(len(labels)):
	 	if (labels[i] not in separated):
	 		separated[labels[i]] = []
	 	separated[labels[i]].append(features[i])

	return separated

@timing
def summarize(features):
	summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*features)]
	return summaries




#-------------------------------------------------------------------------
#   Naive Bayes
#
#   predictions
#-------------------------------------------------------------------------

def getPredictions(summaries, testSet, numFrag):

    size = int(math.ceil(float(len(testSet))/numFrag))
    testSet = [d for d in chunks(testSet, size )]

    from pycompss.api.api import compss_wait_on

    partialResult = [ predict_chunck(summaries, testSet[i])  for i in range(numFrag) ]
    result = mergeReduce(merge_lists, partialResult)
    result = compss_wait_on(result)

    return result

@task(returns=list)
def predict_chunck(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
     	result = predict(summaries, testSet[i])
     	predictions.append(result)
    return predictions

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	#print (probabilities)
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
            probabilities[classValue] *= calculateProbability(x, mean, var)
    return probabilities


def calculateProbability(x, mean, stdev):
    #exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    #prob = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    import functions_naivebayes
    prob = functions_naivebayes.calculateProbability(x,mean,stdev)
    return prob





@task(returns=list)
def merge_lists(list1,list2):
    list1 = list1+list2
    return list1

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Gaussian Naive Bayes PyCOMPSs')
    parser.add_argument('-t', '--TrainSet', required=True, help='path to the train file')
    parser.add_argument('-v', '--TestSet',  required=True, help='path to the test file')
    parser.add_argument('-f', '--Nodes',    type=int,  default=2, required=False, help='Number of nodes')
    parser.add_argument('-o', '--output',   required=False, help='path to the output file')
    arg = vars(parser.parse_args())

    fileTrain = arg['TrainSet']
    fileTest  = arg['TestSet']
    numFrag   = arg['Nodes']
    if arg['output']:
        output_file= arg['output']
    else:
        output_file = ""

    print """Running Gaussian Naive Bayes with the following parameters:
        - Nodes: {}
        - Train Set: {}
        - Test Set: {}\n
        """.format(numFrag,fileTrain,fileTest)


    trainingSet = loadCsv(fileTrain)
    testSet     = loadCsv(fileTest)

    #trainingSet = loadCsv("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/higgs/train_0.012m.csv")
    #testSet     = loadCsv("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/higgs/train_0.012m.csv")
    #dataset  = loadCsv(filename)
    #trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Train={0} and Test={1} rows').format(len(trainingSet), len(testSet))


    x = [i[1] for i in trainingSet]
    y = [i[0] for i in trainingSet]
    #create the model
    summaries = fit(x,y)
    print summaries
	# test model
    test_x = [i[1] for i in testSet]
    test_y = [i[0] for i in testSet]

    predictions = getPredictions(summaries, test_x, numFrag)
    #
    print "predictions"
    #print predictions

    accuracy = getAccuracy(test_y, predictions)
    print('Accuracy: {0}%').format(accuracy)
