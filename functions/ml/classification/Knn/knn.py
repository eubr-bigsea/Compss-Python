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

            :param train: A list of pandas.
            :param K: A number of K nearest neighborhood to take in count.
            :return A model
        """

        train_data = mergeReduce(self.merge_lists, data)

        return train_data


    def transform(self,test_data, train_data, settings, numFrag):
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
        label    = settings['label']
        features = settings['features']
        predictedLabel = settings['new_name'] if 'new_name' in settings else "{}_predited".format(label)
        columns  = label+features
        K = int(settings['K'])


        result = [ self.classifyBlock(  test_data[i],
                                        train_data,
                                        features,
                                        label,
                                        predictedLabel,
                                        K)  for i in range(numFrag) ]


        return result


    def getPopularElement(self,labels,K):
        u, indices = np.unique(labels[0:K], return_inverse=True)
        label = u[np.argmax(np.bincount(indices))]
        return label

    def getKNN(self,neighborhood,K):
        result = [0 for i in range(len(neighborhood))]
        for i in range(len(neighborhood)):
            result[i] = self.getPopularElement(neighborhood[i], K)
        return result

    @task(returns=list, isModifier = False)
    def classifyBlock(self,data,train_data, features, label,predictedLabel,K):

        start=time.time()

        #initalizing variables
        if isinstance(data.iloc[0][features], list):
            numDim = len(data.iloc[0][features])
        else:
            numDim = 1
        print numDim
        sizeTest    = len(data)
        sizeTrain   = len(train_data)
        semi_labels = np.zeros((sizeTest, K+1))
        semi_dist   = np.full( (sizeTest, K+1), float("inf"))
        import functions_knn

        for i_test in range(sizeTest):
            for i_train in range(sizeTrain):
                semi_dist[i_test][K] =  functions_knn.distance(
                                np.array(train_data.iloc[i_train][features]),
                                np.array(data.iloc[i_test][features]),
                                numDim )
                semi_labels[i_test][K] = train_data.iloc[i_train][label]

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

        data[predictedLabel] =  pd.Series(values).values


        end = time.time()
        print "\n[INFO] - ClassifierBlock -> Time elapsed: %.2f seconds\n" % (end-start)

        return data

    @task(returns=list, isModifier = False)
    def merge_lists(self,list1,list2):
        #print "\nmerge_lists\n---\n{}\n---\n{}\n---\n".format(list1,list2)

        if len(list1) == 0:
            result = list2
        elif len(list2) == 0:
            result = list1
        else:
            result = pd.concat([list1,list2], ignore_index=True)
        return  result
