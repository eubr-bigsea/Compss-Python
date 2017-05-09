#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

import time
import math
import numpy as np

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.functions.data    import chunks

def getPopularElement(labels,K):
    #label = max(labels[0:K-2], key = labels.count)
    u, indices = np.unique(labels[0:K], return_inverse=True)
    label = u[np.argmax(np.bincount(indices))]
    return label

def getKNN(neighborhood,K):
    start=time.time()
    result = [0 for i in range(len(neighborhood))]
    for i in range(len(neighborhood)):
        result[i] = getPopularElement(neighborhood[i],K)

    end =time.time()
    print "\n[INFO] - getKNN -> Time elapsed: %.2f seconds\n" % (end-start)
    return result


@task(returns=list)
def classifyBlock(test_data,train_data,numDim,K):
    start=time.time()

    #initalizing variables
    sizeTest    = len(test_data)
    sizeTrain   = len(train_data)
    semi_labels = np.zeros((sizeTest, K+1))                #np.array([[0 for i in range(K+1) ] for i in range(len(test_data))])
    semi_dist   = np.full( (sizeTest, K+1), float("inf"))  #np.array([[float("inf") for i in range(K+1) ] for i in range(len(test_data))])

    import functions_knn
    #calculate part

    for i_test in range(sizeTest):
        for i_train in range(sizeTrain):

            semi_dist  [i_test][K] = functions_knn.distance(train_data[i_train][1:numDim].astype(np.float_), test_data[i_test].astype(np.float_), numDim)
            semi_labels[i_test][K] = train_data[i_train][0]

            j=K
            while(j>0):
                if(semi_dist[i_test][j] < semi_dist[i_test][j-1]):
                    tmp_label = semi_labels[i_test][j]
                    semi_labels[i_test][j]   = semi_labels[i_test][j-1]
                    semi_labels[i_test][j-1] = tmp_label

                    tmp_dist = semi_dist[i_test][j];
                    semi_dist[i_test][j]    = semi_dist[i_test][j-1];
                    semi_dist[i_test][j-1]  = tmp_dist;
                j-=1

    print "exiting"
    #Choose part
    result_partial= getKNN(semi_labels,K)

    end = time.time()
    print "\n[INFO] - ClassifierBlock -> Time elapsed: %.2f seconds\n" % (end-start)

    return result_partial

# @task(returns=list)
# def merge_lists(list1,list2):
#     list1 = list1+list2
#     return list1

def knn(train_data,test_data, K, numFrag, output_file):
    """
        K-Nearest Neighbor is an non parametric lazy learning algorithm.
        What this means is that it does not use the training data points to
        do any generalization.  In other words, there is no explicit training
        phase. More exactly, all the training data is needed during the
        testing phase. The Classification is computed from a simple majority
        vote of the nearest neighbors of each point present in the training set.

        :param train: A np.array already merged. Each line row in
                      this format [label,[features]]
        :param test_data: A np.array already splitted.
                          Each line row in this format [features]
        :param K: A number of K nearest neighborhood to take in count.
        :param numFrag: number of fragments. (I can remove that later)
        :param output_file: List of name to save the output or a empty list if you
                            don't want to write the output in a file.
        :return: list of labels predicted.
    """

    numDim = len(train_data)-1
    from pycompss.api.api import compss_wait_on

    partialResult = [ classifyBlock(test_data[i],train_data,numDim,K)  for i in range(numFrag) ]
    #result = mergeReduce(merge_lists, partialResult)
    #result = compss_wait_on(result)
    if len(output_file) > 1:
        for p in range(len(partialResult)):
            filename= output_file+"_part"+str(p)
            f = open(filename,"w")
            f.close()
            #savePredictionToFile(partialResult[p], filename)
        return []
    else:
        partialResult = compss_wait_on(partialResult)
        return partialResult
