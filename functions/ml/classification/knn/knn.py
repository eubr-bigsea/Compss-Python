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


class KNN(object):

    def fit(self,train_data,K):
        """
            K-Nearest Neighbor is an non parametric lazy learning algorithm.
            What this means is that it does not use the training data points to
            do any generalization.  In other words, there is no explicit training
            phase. More exactly, all the training data is needed during the
            testing phase.

            :param train: A np.array already merged. Each line row in
                          this format [label,[features]]
            :param K: A number of K nearest neighborhood to take in count.
            :return A model
        """

        return [train_data,K]


    def transform(self,test_data,model, numFrag, output_file):
        """
            K-Nearest Neighbor: The Classification is computed from a simple
            majority vote of the nearest neighbors of each point present in the
            training set.


            :param test_data: A np.array already splitted.
                              Each line row in this format [features]
            :param numFrag:   number of fragments. (I can remove that later)
            :param output_file: List of name to save the output or a empty list
                                if you don't want to write the output in a file.
            :return A list of labels predicted.
        """

        from pycompss.api.api import compss_wait_on

        partialResult = [ self.classifyBlock(test_data[i],model[0],
                                        model[1])  for i in range(numFrag) ]

        if len(output_file) > 1:
            for p in range(len(partialResult)):
                filename= output_file+"_part"+str(p)
                f = open(filename,"w")
                f.close()
                savePredictionToFile(partialResult[p], filename)
            return []
        else:
            result = [ mergeReduce(self.merge_lists, partialResult)]
            result  = compss_wait_on(result)
            return result[0]


    def getPopularElement(self,labels,K):

        u, indices = np.unique(labels[0:K], return_inverse=True)
        label = u[np.argmax(np.bincount(indices))]
        return label

    def getKNN(self,neighborhood,K):
        start=time.time()
        result = [0 for i in range(len(neighborhood))]
        for i in range(len(neighborhood)):
            result[i] = self.getPopularElement(neighborhood[i],K)

        end =time.time()
        print "\n[INFO] - getKNN -> Time elapsed: %.2f seconds\n" % (end-start)
        return result

    @task(returns=list, isModifier = False)
    def classifyBlock(self,test_data,train_data,K):

        start=time.time()

        numDim = len(train_data[0])-1

        #initalizing variables
        sizeTest    = len(test_data)
        sizeTrain   = len(train_data)
        semi_labels = np.zeros((sizeTest, K+1))
        semi_dist   = np.full( (sizeTest, K+1), float("inf"))
        import functions_knn

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


        result_partial= self.getKNN(semi_labels,K)

        end = time.time()
        print "\n[INFO] - ClassifierBlock -> Time elapsed: %.2f seconds\n" % (end-start)

        return result_partial

    @task(returns=list, isModifier = False)
    def merge_lists(self,list1,list2):
        return list1+list2
