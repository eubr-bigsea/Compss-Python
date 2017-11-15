#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

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

        Methods:
            - fit()
            - fit_transform()
            - transform()


    """

    def fit(self, data, settings, numFrag):
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

        train_data = [[] for f in range(numFrag)]
        for f in range(numFrag):
            train_data[f] = createModel(data[f], col_label, col_features)
        train_data = mergeReduce(merge_lists, train_data)

        model = {}
        model['algorithm'] = 'KNN'
        model['model'] = train_data
        return model


    def fit_transform(self, data, settings, numFrag):
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
        predCol = settings.get('predCol', "predited")
        K = settings.get('K', 1)


        p_model = [ createModel(data[f],
                                col_label,
                                col_features) for f in range(numFrag)]

        model = mergeReduce(merge_lists, p_model)


        result = [[] for i in range(numFrag)]
        for i in range(numFrag):
            result[i] = classifyBlock(data[i], model, col_features, predCol, K)

        return result

    def transform(self, data, model, settings, numFrag):
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

        if 'features' not in settings:
           raise Exception("You must inform the at least the `features` field.")

        if model.get('algorithm','null') != 'KNN':
            raise Exception("You must inform a valid model.")

        model = model['model']

        col_features = settings['features']
        predCol = settings.get('predCol', "predited")
        K = settings.get('K', 1)

        result = [[] for i in range(numFrag)]
        for i in range(numFrag):
            result[i] = classifyBlock(data[i], model, col_features, predCol, K)


        return result


@task(returns=list)
def createModel(df,label,features):
    labels = df[label].values
    feature = np.array(df[features].values.tolist())

    # df.rename(columns={label: 'label', features: 'features'}, inplace=True)
    return [labels, feature] #df

@task(returns=list)
def merge_lists(list1, list2):
    # pd.concat([list1,list2], ignore_index=True)
    l1,f1 = list1
    l2,f2 = list2

    if len(l1) != 0:
        if  len(l2) != 0:
            result1 = np.concatenate((l1,l2), axis=0)
            result2 = np.concatenate((f1,f2), axis=0)
            return  [result1, result2]
        else:
            return list1
    else:
        return list2



@task(returns=list)
def classifyBlock(data, model, col_features, predCol, K):

    sizeTest    = len(data)
    sizeTrain   = len(model[0])

    #if this frame is empty, do nothing
    if sizeTest == 0:
        data[predCol] = np.nan
        return data

    #initalizing variables
    sample = data.iloc[0][col_features]
    if isinstance(sample, list):
        numDim = len(sample)
    else:
        numDim = 1

    semi_labels = [ [ 0 for i in range(K)] for j in range(sizeTest)] #without type

    import functions_knn

    for i_test in range(sizeTest):
        semi_dist   = np.empty(sizeTrain)
        for i_train in range(sizeTrain):
            semi_dist[i_train] =   functions_knn.distance( #np.array(train_data.iloc[i_train]['features']).astype(float)
                            model[1][i_train],
                            np.array(data.iloc[i_test][col_features]),
                            numDim )
            # semi_dist[i_train] =  np.linalg.norm(model[1][i_train]-np.array(data.iloc[i_test][col_features]))
        inds = semi_dist.argsort()
        semi_labels[i_test] =  model[0][inds][0:K]

    values = getKNN(semi_labels, K)
    data[predCol] =  pd.Series(values).values

    return data


def getPopularElement(labels,K):
    u, indices = np.unique(labels[0:K], return_inverse=True)
    label = u[np.argmax(np.bincount(indices))]
    return label

def getKNN(neighborhood,K):
    result = [0 for i in range(len(neighborhood))]
    for i in range(len(neighborhood)):
        result[i] = getPopularElement(neighborhood[i], K)
    return result
