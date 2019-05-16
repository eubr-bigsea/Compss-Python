#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF, generate_info
from ddf_library.ddf_model import ModelDDF

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on

import math
import numpy as np
import pandas as pd


class KNearestNeighbors(ModelDDF):

    """K-Nearest Neighbor is a algorithm used that can be used for both
    classification and regression predictive problems. In a classification, the
    algorithm computes from a simple majority vote of the K nearest neighbors
    of each point present in the training set. The choice of the parameter
    K is very crucial in this algorithm, and depends on data set.
    However, values of one or tree is more common.

    :Example:

    >>> knn = KNearestNeighbors(feature_col='features',
    >>>                         label_col='label', k=1).fit(ddf1)
    >>> ddf2 = knn.transform(ddf1)
    """

    def __init__(self, feature_col, label_col, k=3):
        """
        :param feature_col: Feature column name;
        :param label_col: Label column name;
        :param k: Number of nearest neighbors to majority vote;
        """
        super(KNearestNeighbors, self).__init__()

        if not isinstance(feature_col, list):
            feature_col = [feature_col]

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['label_col'] = label_col
        self.settings['k'] = k

        self.model = []
        self.name = 'KNearestNeighbors'

    def fit(self, data):
        """
        Fit the model.

        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        col_label = self.settings['label_col']
        col_feature = self.settings['feature_col']
        k = self.settings['k']

        train_data = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            train_data[f] = _knn_create_model(df[f], col_label,
                                              col_feature, nfrag, k)
        model = merge_reduce(merge_lists, train_data)

        self.model = [compss_wait_on(model)]
        return self

    def fit_transform(self, data, pred_col='prediction_kNN'):
        """
        Fit the model and transform.

        :param data: DDF
        :param pred_col: Output prediction name (default, *'prediction_kNN'*);
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, pred_col=pred_col)
        return ddf

    def transform(self, data, feature_col=None, pred_col='prediction_kNN'):
        """

        :param data: DDF
        :param feature_col: Feature column name
        :param pred_col: Output prediction name (default, *'prediction_kNN'*);
        :return: DDF
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        task_list = data.task_list
        settings = self.settings.copy()
        settings['pred_col'] = pred_col
        settings['model'] = self.model[0].copy()
        if feature_col is not None:
            settings['feature_col'] = feature_col

        def task_transform_knn(df, params):
            return _knn_classify_block(df, params)

        uuid_key = self._ddf_add_task(task_name='task_transform_nb',
                                      opt=self.OPT_SERIAL,
                                      function=[task_transform_knn,
                                                settings],
                                      parent=[data.last_uuid])

        return DDF(task_list=task_list, last_uuid=uuid_key)


@task(returns=1)
def _knn_create_model(df, label, features, nfrag, k):
    """Create a partial model based in the selected columns."""
    labels = df[label].values
    feature = df[features].values
    return [[labels], [feature], 0, nfrag, k]


@task(returns=1)
def merge_lists(list1, list2):
    """Merge all elements in an unique DataFrame to be part of a knn model."""
    l1, f1, i1, nfrag, k = list1
    l2, f2, i2, _, _ = list2

    f1 = f1 + f2
    l1 = l1 + l2

    i = i1+i2 + 1
    if i == (nfrag - 1):
        l1 = np.concatenate(l1, axis=0)
        f1 = np.concatenate(f1, axis=0)

        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(f1, l1)
        return [neigh]

    return [l1, f1, i, nfrag, k]


def _knn_classify_block(data, settings):
    """Perform a partial classification."""
    col_features = settings['feature_col']
    pred_col = settings.get('pred_col', "prediction_kNN")
    model = settings['model'][0]
    frag = settings['id_frag']

    if pred_col in data.columns:
        data.drop([pred_col], axis=1, inplace=True)

    if len(data) > 0:
        data[pred_col] = model.predict(data[col_features])
    else:
        data[pred_col] = np.nan

    info = generate_info(data, frag)
    return data, info
