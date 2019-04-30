#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ddf_library.ddf import DDF
import numpy as np


def base():
    # loading a csv file from HDFS
    ddf = DDF().load_text('/iris-dataset.csv', num_of_parts=4, sep=',',
                          dtype={'class': np.dtype('O'),  # string
                                 'sepal_length': np.float64,
                                 'sepal_width':  np.float64,
                                 'petal_length': np.float64,
                                 'petal_width':  np.float64
                                 }) \
        .replace({'Iris-setosa': 1.0,
                  'Iris-versicolor': -1.0,
                  'Iris-virginica': 1.0}, subset=['class']).cast('class',
                                                                 'integer')

    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # scaling using StandardScaler
    from ddf_library.functions.ml.feature import StandardScaler
    ddf = StandardScaler(input_col=cols).fit_transform(ddf)

    # splitting 25% to use as training set and 75% as test
    ddf_train, ddf_test = ddf.split(0.50)

    return ddf_train, ddf_test


def gaussian(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    from ddf_library.functions.ml.classification import GaussianNB
    nb = GaussianNB(feature_col=cols, label_col='class')\
        .fit(ddf_train)
    nb.save_model('/gaussian_nb')  # save this fitted model in HDFS
    ddf_pred = nb.transform(ddf_test)
    return ddf_pred


def knn(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    from ddf_library.functions.ml.classification import KNearestNeighbors
    knn = KNearestNeighbors(k=1, feature_col=cols, label_col='class')\
        .fit(ddf_train)
    knn.save_model('/knn')
    ddf_pred = knn.transform(ddf_test, pred_col='prediction')

    return ddf_pred


def logistic_regression(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    from ddf_library.functions.ml.classification import LogisticRegression
    logr = LogisticRegression(feature_col=cols, label_col='class',
                              max_iters=10, regularization=0.1,
                              threshold=0.1).fit(ddf_train)
    logr.save_model('/logistic_regression')

    ddf_pred = logr.transform(ddf_test, pred_col='prediction')

    return ddf_pred


def svm(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    from ddf_library.functions.ml.classification import SVM
    svm = SVM(feature_col=cols, label_col='class', max_iters=100,
              coef_lambda=1, coef_lr=0.01).fit(ddf_train)
    svm.save_model('/svm')
    ddf_pred = svm.transform(ddf_test, pred_col='prediction')

    return ddf_pred


def binary_evaluator(ddf_pred, pred_col):
    from ddf_library.functions.ml.evaluation import BinaryClassificationMetrics

    metrics_bin = BinaryClassificationMetrics(label_col='class',
                                              true_label=1,
                                              pred_col=pred_col,
                                              ddf_var=ddf_pred)

    # Get some metrics
    print("Binary Metrics:\n", metrics_bin.get_metrics(), '\n',
          metrics_bin.confusion_matrix)


def multi_label_evaluator(ddf_pred, pred_col):
    from ddf_library.functions.ml.evaluation import MultilabelMetrics

    metrics_multi = MultilabelMetrics(label_col='class',
                                      pred_col=pred_col,
                                      ddf_var=ddf_pred)

    print("Multilabel Metrics:\n",
          metrics_multi.get_metrics(), '\n',
          metrics_multi.confusion_matrix, '\n',
          metrics_multi.precision_recall)


def ml_classifiers_part2(ddf):

    # Loading previous fitted ml models
    from ddf_library.functions.ml.classification import GaussianNB,\
        KNearestNeighbors, LogisticRegression,  SVM

    nb = GaussianNB(feature_col='features', label_col='label') \
        .load_model('/gaussian_nb')
    knn = KNearestNeighbors(k=1, feature_col='features', label_col='label') \
        .load_model('/knn')
    logr = LogisticRegression(feature_col='features', label_col='label') \
        .load_model('/logistic_regression')
    svm = SVM(feature_col='features', label_col='label')\
        .load_model('/svm')

    ddf = knn.transform(ddf)
    ddf = svm.transform(ddf)
    ddf = logr.transform(ddf)
    df = nb.transform(ddf).to_df()

    print('All classifiers:\n', df)


if __name__ == '__main__':
    print("_____Testing Machine Learning Classifiers_____")

    ddf_train, ddf_test = base()
    # ddf_pred = gaussian(ddf_train, ddf_test)
    # ddf_pred = knn(ddf_train, ddf_test)
    # ddf_pred = logistic_regression(ddf_train, ddf_test)
    ddf_pred = svm(ddf_train, ddf_test)
    # ddf_pred.show()
    binary_evaluator(ddf_pred, 'prediction')
    # ml_classifiers_part2(ddf_pred)
