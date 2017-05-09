#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import time
import math
import numpy as np

def chunks(l, n, balanced=False):
    if not balanced or not len(l) % n:
        for i in xrange(0, len(l), n):
            yield l[i:i + n]
    else:
        rest = len(l) % n
        start = 0
        while rest:
            yield l[start: start+n+1]
            rest -= 1
            start += n+1
        for i in xrange(start, len(l), n):
            yield l[i:i + n]


@task(returns=list, priority=True)
def reduceCentersTask(a, b):
    #return [a,b]
    return [a[0]+b[0],a[1]+b[1]]


def distance(feature1,feature2):
    distance = 0
    for i,j in zip(feature1,feature2):
        distance += (i-j)*(i-j)

    return  math.sqrt(distance)

def insertionSort(A):
    i=len(A)-1
    x_l = A[i][0]
    x_d = A[i][1]
    j = i-1
    while j>=0 and x_d<A[j][1]:
        A[j+1][1] = A[j][1]
        A[j+1][0] = A[j][0]
        j=j-1
    A[j+1][1] = x_d
    A[j+1][0] = x_l

    return A

@task(returns=list)
def classifyBlock(test_data,train_data,K):
    result_partial=[]

    for feature1 in test_data:
        bestKPoints= [[-1,99999999999] for i in range(K+1)] #label,dist
        for feature2 in train_data:
            dist = distance(feature1[1],feature2[1])
            bestKPoints[K] = [feature2[0], dist]
            bestKPoints = insertionSort(bestKPoints)

        result_partial.append(bestKPoints[0:K])
    return result_partial

@task(returns=list)
def mergerPartialResults(partialResult,partialResult2,K):
    result =[]
    for i in xrange(0,len(partialResult)):
        tmp = []
        i_left = 0
        i_right = 0

        for t in range(K):
            if partialResult[i][i_left][1]<=partialResult2[i][i_right][1]:
                tmp.append(partialResult[i][i_left][0])
                i_left+=1
            else:
                tmp.append(partialResult2[i][i_right][0])
                i_right+=1

        label = max(tmp, key = tmp.count)
        result.append(label)

    return result

@task(returns=list)
def evaluate(partialResult,test_data):

    correct=0
    total = len(test_data)

    for p,t in zip(partialResult, test_data):
        if p == t[0]:
            correct+=1

    return [correct, total]



def knn(train_data,test_data, K, numFrag):
    """ Knn:

    :param data: data
    :param K: num of K nearest neighborhood to take in count
    :param numFrag: num fragments, if -1 data is considered chunked
    :return: list of labels predicted
    """
    # Data is already fragmented
    if numFrag == -1:
        numFrag = len(train_data)
    else:
        train_data = [d for d in chunks(train_data, len(train_data)/numFrag)]

    if numFrag == -1:
        numFrag = len(test_data)
    else:
        test_data = [d for d in chunks(test_data, len(test_data)/numFrag)]



    result=[]
    from pycompss.api.api import compss_wait_on

    partialResult = [classifyBlock(test_data[i],train_data[j],K)  for j in range(numFrag) for i in range(numFrag) ]
    merged_partialResult = [mergerPartialResults(partialResult[i],partialResult[i+numFrag],K)  for i in range(numFrag) ]
    #numcorrects   = [evaluate(merged_partialResult[i],test_data[i])   for i in range(numFrag)]

    #result =  mergeReduce(reduceCentersTask, numcorrects)
    #result =  compss_wait_on(result)
    merged_partialResult = compss_wait_on(merged_partialResult)

    #labels_Result = [] # do colaest
    #for p in merged_partialResult:
    #    labels_Result += p

    return result, merged_partialResult #labels_Result

def read_file_vector(name,separator):
    row = []

    for line in open(name,"r"):
        col = line.split(separator)
        label = -1
        features =[]

        for i in xrange(0,len(col)):
            if i is 0: # change
                label = float(col[i])
            else:
                features.append( float(col[i]))
        row.append([label,features])

    return row


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='KNN PyCOMPSs')
    parser.add_argument('-t', '--TrainSet', required=True, help='path to the train file')
    parser.add_argument('-v', '--TestSet',  required=True, help='path to the test file')
    parser.add_argument('-f', '--Nodes',    type=int,  default=2, required=False, help='Number of nodes')
    parser.add_argument('-k', '--K',        type=int,  default=1, required=False, help='Number of nearest neighborhood')
    arg = vars(parser.parse_args())
	#parser.add_argument('-o', '--output', required=True, help='path to the output file')

    fileTrain = arg['TrainSet']
    fileTest = arg['TestSet']
    k = arg['K']
    numFrag = arg['Nodes']

    separator = ","

    start=time.time()
    train_data = read_file_vector(fileTrain,separator)
    end=time.time()
    print "[INFO] - read_file_vector -> ", end-start
    start=time.time()
    test_data = read_file_vector(fileTest,separator)
    end=time.time()
    print "[INFO] - read_file_vector -> ", end-start
    start=time.time()
    result_labels, partialResult = knn(train_data,test_data, k, numFrag)
    end=time.time()
    print "[INFO] - read_file_vector -> ", end-start
    #print "Acurracy: {}".format(float(result_labels[0])/result_labels[1])

    #print partialResult
    print "END"
