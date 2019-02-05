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
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on


import uuid




class KNearestNeighbors(object):

    """KNN's Methods.

    - fit(): Create a model based in an dataset.
    - transform(): Predict a dataset based in the model created.
    """

    def __init__(self, feature_col, label_col, pred_col=None, k=3):
        if not feature_col:
            raise Exception("You must inform the `features` field.")

        if not label_col:
            raise Exception("You must inform the `label` field.")

        if not pred_col:
            pred_col = 'prediction_kNN'

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['pred_col'] = pred_col
        self.settings['k'] = k

        self.model = None

    def fit(self, data):
        """

        :param data: DDF
        :return: trained model
        """

        df = data.partitions[0]

        nfrag = len(df)
        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']
        train_data = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            train_data[f] = create_model(df[f], col_label, col_feature)
        model = merge_reduce(merge_lists, train_data)

        self.model = compss_wait_on(model)
        return self

    def transform(self, data):
        """

        :param data: DDF
        :return:
        """

        if not self.model:
            raise Exception("Model is not fitted.")

        model = self.model

        def task_transform_knn(df, params):
            return _classify_block_(df, model, params)

        uuid_key = str(uuid.uuid4())
        COMPSsContext.tasks_map[uuid_key] = {'name': 'task_transform_knn',
                                             'status': 'WAIT', 'lazy': True,
                                             'function': [task_transform_knn,
                                                          self.settings],
                                             'parent': [data.last_uuid],
                                             'output': 1, 'input': 1}

        data.set_n_input(uuid_key, data.settings['input'])
        return DDF(data.partitions, data.task_list, uuid_key)


@task(returns=list)
def create_model(df, label, features):
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


def _classify_block_(data, model, settings):
    """Perform a partial classification."""
    col_features = settings['feature_col']
    pred_col = settings.get('pred_col', "prediction_kNN")
    K = settings.get('k', 3)

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
