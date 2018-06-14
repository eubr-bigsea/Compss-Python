#!/usr/bin/python
# -*- coding: utf-8 -*-
"""K-Nearest Neighbor.

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
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import numpy as np
import pandas as pd
from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce
from pycompss.api.api import compss_wait_on


class KNN(object):
    """KNN's Methods.

    - fit(): Create a model based in an dataset.
    - transform(): Predict a dataset based in the model created.
    """

    def fit(self, data, settings, nfrag):
        """Fit.

        :param data: A list with nfrag pandas's dataframe used to
            training the model.
        :param settings: A dictionary that contains:
            - K: Number of K nearest neighborhood to take in count.
            - features: Column name of the features in the training data;
            - label: Column name of the labels in the training data;
        :param nfrag: A number of fragments;
        :return: The model created (which is a pandas dataframe).
        """
        col_label = settings['label']
        col_features = settings['features']

        train_data = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            train_data[f] = createModel(data[f], col_label, col_features)
        train_data = mergeReduce(merge_lists, train_data)

        train_data = compss_wait_on(train_data)
        model = dict()
        model['algorithm'] = 'KNN'
        model['model'] = train_data
        return model

    def transform(self, data, model, settings, nfrag):
        """Transform.

        :param data: A list with nfrag pandas's dataframe that will
            be predicted.
        :param model: The KNN model created;
        :param settings: A dictionary that contains:
            - K: Number of K nearest neighborhood to take in count.
            - features: Column name of the features in the test data;
            - predCol: Alias to the new column with the labels predicted;
        :param nfrag: A number of fragments;
        :return: The prediction (in the same input format).
        """
        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        if model.get('algorithm', 'null') != 'KNN':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = [[] for _ in range(nfrag)]
        for i in range(nfrag):
            result[i] = _classify_block(data[i], model, settings)
        return result

    def transform_serial(self, data, model, settings):

        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        if model.get('algorithm', 'null') != 'KNN':
            raise Exception("You must inform a valid model.")

        model = model['model']

        result = _classify_block_(data, model, settings)
        return result


@task(returns=list)
def createModel(df, label, features):
    """Create a partial model based in the selected columns."""
    labels = df[label].values
    feature = np.array(df[features].values.tolist())
    return [labels, feature]


@task(returns=list)
def merge_lists(list1, list2):
    """Merge all elements in an unique dataframe to be part of a knn model."""
    l1, f1 = list1
    l2, f2 = list2

    if len(l1) != 0:
        if len(l2) != 0:
            result1 = np.concatenate((l1, l2), axis=0)
            result2 = np.concatenate((f1, f2), axis=0)
            return [result1, result2]
        else:
            return list1
    else:
        return list2


@task(returns=list)
def _classify_block(data, model, settings):
    return _classify_block_(data, model, settings)


def _classify_block_(data, model, settings):
    """Perform a partial classification."""
    col_features = settings['features']
    pred_col = settings.get('predCol', "predited")
    K = settings.get('K', 1)

    sizeTest = len(data)
    if sizeTest == 0:
        data[pred_col] = np.nan
        return data

    # initalizing variables
    sample = data.iloc[0][col_features]
    if isinstance(sample, list):
        numDim = len(sample)
    else:
        numDim = 1

    semi_labels = [[0 for _ in range(K)] for _ in range(sizeTest)]

    import functions_knn
    from timeit import default_timer as timer
    start = timer()

    semi_labels = functions_knn.dist2all(
                  model[1], np.array(data.iloc[:][col_features].tolist()),
                  numDim, K, semi_labels, model[0])
    end = timer()
    print "{0:.2e}".format(end - start)
    values = get_knn(semi_labels)
    data[pred_col] = pd.Series(values).values

    return data


def get_knn(neighborhood):
    """Finding the most frequent label."""
    result = np.zeros(len(neighborhood)).tolist()
    for i in range(len(neighborhood)):
        labels = neighborhood[i].tolist()
        result[i] = \
            max(map(lambda val: (labels.count(val), val), set(labels)))[1]
    return result
