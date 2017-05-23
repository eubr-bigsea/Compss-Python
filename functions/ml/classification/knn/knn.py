#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

import time
import math
import numpy as np
import pandas as pd

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.functions.data    import chunks
from pycompss.api.api import compss_wait_on

class KNN(object):

    def fit(self,data,settings,numFrag):
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

        labels   = settings['labels']
        features = settings['features']
        columns  = labels+features

        partial  = [self.format_data(data[i],columns) for i in range(numFrag)]
        train_data = mergeReduce(self.merge_lists, partial)
        #columns = settings['labels']+settings['features']
        #data = compss_wait_on(data)
        #partial = [self.format_data(data[i],columns) for i in range(numFrag)]
        #train_data = mergeReduce(self.merge_lists, partial)
        #train_data  = compss_wait_on(train_data)


        return train_data

    @task(returns=list, isModifier = False)
    def format_data(self,data,columns):
        return  data[columns].values

    def transform(self,test_data,settings, numFrag):
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
        labels   = settings['labels']
        features = settings['features']
        columns  = labels+features
        train_data = settings['model']
        K = int(settings['K'])


        partialResult = [ self.classifyBlock(test_data[i], train_data,
                                             features, labels,K)  for i in range(numFrag) ]

        #result = [ mergeReduce(self.merge_lists, partialResult)]
        #result  = compss_wait_on(result)
        return partialResult#result[0]


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
    def classifyBlock(self,data,train_data, features, label,K):

        start=time.time()

        test_data = np.array(data[features].values)
        numDim = len(test_data[0])

        print numDim
        print train_data
        print test_data
        print "----"
        #initalizing variables
        sizeTest    = len(test_data)
        sizeTrain   = len(train_data)
        semi_labels = np.zeros((sizeTest, K+1))
        semi_dist   = np.full( (sizeTest, K+1), float("inf"))
        import functions_knn

        for i_test in range(sizeTest):
            for i_train in range(sizeTrain):
                #print train_data[i_train][1:numDim].astype(np.float_)
                #print test_data[i_test].astype(np.float_)

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


        values= self.getKNN(semi_labels,K)
        new_column = "_".join(i for i in label) +"_predited"
        data[new_column] =  pd.Series(values).values


        end = time.time()
        print "\n[INFO] - ClassifierBlock -> Time elapsed: %.2f seconds\n" % (end-start)

        return data

    @task(returns=list, isModifier = False)
    def merge_lists(self,list1,list2):
        print "\nmerge_lists\n---\n{}\n---\n{}\n---\n".format(list1,list2)

        if len(list1) == 0:
            result = list2
        elif len(list2) == 0:
            result = list1
        else:
            result = np.concatenate((list1, list1), axis=0)
        return  result
