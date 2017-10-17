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

class KNN(object):

    """
        K-Nearest Neighbor:

        K-Nearest Neighbor is a algorithm used that can be used for both
        classification and regression predictive problems. However, it is
        more widely used in classification problems. Is a non parametric
        lazy learning algorithm. Meaning that it does not use the training
        data points to do any generalization.  In other words, there is no
        explicit training phase. More precisely, all the training data is
        needed during the testing phase.

        To do a classification, the algorithm computes from a simple majority
        vote of the K nearest neighbors of each point present in the training
        set. The choice of the parameter K is very crucial in this algorithm,
        and depends on the dataset. However, values of one or tree is more
        commom.

    """

    def fit(self,data,settings,numFrag):
        """
            fit()

            :param data:     A list with numFrag pandas's dataframe used to
                             training the model.
            :param settings: A dictionary that contains:
                - K:  		 Number of K nearest neighborhood to take in count.
                - features:  Column name of the features in the training data;
                - label:     Column name of the labels   in the training data;
            :param numFrag:  A number of fragments;
            :return:         The model created (which is a pandas dataframe).
        """
        col_label    = settings['label']
        col_features = settings['features']

        data     = [self.createModel(data[f],col_label,col_features) for f in range(numFrag)]
        train_data = mergeReduce(self.merge_lists, data)

        return train_data


    def fit_transform(self,data, settings, numFrag):
        """
            fit_transform():

            :param data:     A list with numFrag pandas's dataframe used to
                             training the model and to classify it.
            :param settings: A dictionary that contains:
             - K:  			 Number of K nearest neighborhood to take in count.
                             (default, 1)
             - features: 	 Field of the features in the training/test data;
             - label:        Field of the labels   in the training/test data;
             - predCol:      Alias to the new column with the labels predicted;
            :param numFrag:  A number of fragments;
            :return:         The prediction (in the same input format) and the
                             model (which is a pandas dataframe).
        """

        col_label    = settings['label']
        col_features = settings['features']
        predCol = settings.get('predCol', "{}_predited".format(col_label))
        K = settings.get('K', 1)


        p_model = [ self.createModel(data[f],
                                    col_label,
                                    col_features) for f in range(numFrag)]

        model = mergeReduce(self.merge_lists, p_model)

        result = [ self.classifyBlock(  data[i],
                                        model,
                                        col_features,
                                        predCol,
                                        K)  for i in range(numFrag) ]


        return result

    def transform(self,test_data, model, settings, numFrag):
        """
            transform():

            :param data:     A list with numFrag pandas's dataframe that will
                             be predicted.
            :param model:	 The KNN model created;
            :param settings: A dictionary that contains:
                - K:     	 Number of K nearest neighborhood to take in count.
                - features:  Column name of the features in the test data;
                - predCol:   Alias to the new column with the labels predicted;
            :param numFrag:  A number of fragments;
            :return:         The prediction (in the same input format).
        """
        col_features = settings['features']
        predCol = settings.get('predCol', "predited")
        K = settings.get('K', 1)


        result = [ self.classifyBlock(  test_data[i],
                                        model,
                                        col_features,
                                        predCol,
                                        K)  for i in range(numFrag) ]


        return result


    @task(returns=list, isModifier = False)
    def createModel(self,data,label,features):
        model = pd.DataFrame([])
        model['label']    = data[label]
        model['features'] = data[features]
        return model



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
    def classifyBlock(self,data,train_data, col_features, predCol,K):

        start=time.time()
        sizeTest    = len(data)
        sizeTrain   = len(train_data)

        #if this frame is empty, do nothing
        if sizeTest==0:
            data[predCol] = np.nan
            return data



        #initalizing variables
        if isinstance(data.iloc[0][col_features], list):
            numDim = len(data.iloc[0][col_features])
        else:
            numDim = 1


        semi_labels = [ [ 0 for i in range(K+1)] for j in range(sizeTest)]
        semi_dist   = np.full( (sizeTest, K+1), float("inf"))
        import functions_knn

        for i_test in range(sizeTest):
            for i_train in range(sizeTrain):
                semi_dist[i_test][K] =  functions_knn.distance(
                                np.array(train_data.iloc[i_train]['features']),
                                np.array(data.iloc[i_test][col_features]),
                                numDim )
                semi_labels[i_test][K] = train_data.iloc[i_train]['label']

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


        values = self.getKNN(semi_labels,K)
        data[predCol] =  pd.Series(values).values

        end = time.time()
        print "[INFO] - ClassifierBlock: Time elapsed: %.2f seconds\n" \
                % (end-start)

        return data

    @task(returns=list, isModifier = False)
    def merge_lists(self,list1,list2):
        result = pd.concat([list1,list2], ignore_index=True)
        return  result
