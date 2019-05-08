#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ddf_library.ddf import DDF
import numpy as np


def base():
    # loading a csv file from HDFS
    ddf = DDF().load_text("/iris-dataset.csv", num_of_parts=4, sep=',',
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

    # splitting 50% to use as training set and 50% as test
    train, test = ddf.split(0.50)

    return train, test


def gaussian_classifier(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    from ddf_library.functions.ml.classification import GaussianNB
    nb = GaussianNB(feature_col=cols, label_col='class')\
        .fit(ddf_train)
    nb.save_model('/gaussian_nb')  # save this fitted model in HDFS
    out_data = nb.transform(ddf_test, pred_col='prediction')
    return out_data


def knn_classifier(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    from ddf_library.functions.ml.classification import KNearestNeighbors
    knn = KNearestNeighbors(k=1, feature_col=cols, label_col='class')\
        .fit(ddf_train)
    knn.save_model('/knn')
    out_data = knn.transform(ddf_test, pred_col='prediction')

    return out_data


def logistic_regression_classifier(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    from ddf_library.functions.ml.classification import LogisticRegression
    logr = LogisticRegression(feature_col=cols, label_col='class',
                              max_iter=10, regularization=0.1,
                              threshold=0.1).fit(ddf_train)
    logr.save_model('/logistic_regression')

    out_data = logr.transform(ddf_test, pred_col='prediction')

    return out_data


def svm_classifier(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    from ddf_library.functions.ml.classification import SVM
    svm = SVM(feature_col=cols, label_col='class', max_iter=100,
              coef_lambda=1, coef_lr=0.01).fit(ddf_train)
    svm.save_model('/svm')
    out_data = svm.transform(ddf_test, pred_col='prediction')

    return out_data


def binary_evaluator(data, pred_col):
    from ddf_library.functions.ml.evaluation import BinaryClassificationMetrics

    metrics_bin = BinaryClassificationMetrics(label_col='class',
                                              true_label=1,
                                              pred_col=pred_col,
                                              ddf_var=data)

    # Get some metrics
    print("Binary Metrics:\n", metrics_bin.get_metrics(), '\n',
          metrics_bin.confusion_matrix)


def multi_label_evaluator(data, pred_col):
    from ddf_library.functions.ml.evaluation import MultilabelMetrics

    metrics_multi = MultilabelMetrics(label_col='class',
                                      pred_col=pred_col,
                                      ddf_var=data)

    print("Multi label Metrics:\n",
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

    ddf1, ddf2 = base()
    ddf_pred = gaussian_classifier(ddf1, ddf2)
    # ddf_pred = knn_classifier(ddf1, ddf2)
    # ddf_pred = logistic_regression_classifier(ddf1, ddf2)
    # ddf_pred = svm_classifier(ddf1, ddf2)
    # ddf_pred.show()
    binary_evaluator(ddf_pred, 'prediction')
    # ml_classifiers_part2(ddf_pred)
