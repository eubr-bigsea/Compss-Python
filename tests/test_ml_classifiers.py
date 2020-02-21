#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ddf_library.context import COMPSsContext
from ddf_library.types import DecimalType, StringType, IntegerType
import numpy as np


def base(cc):
    # loading a csv file from HDFS
    ddf = cc\
        .read.csv("hdfs://localhost:9000/iris-dataset.csv", sep=',',
                  schema={'sepal_length': DecimalType,
                          'sepal_width': DecimalType,
                          'petal_length': DecimalType,
                          'petal_width': DecimalType,
                          'class': StringType}) \
        .replace({'Iris-setosa': 1.0,
                  'Iris-versicolor': -1.0,
                  'Iris-virginica': 1.0}, subset=['class'])\
        .cast('class', IntegerType)

    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    from ddf_library.functions.ml.feature import StandardScaler
    ddf = StandardScaler().fit_transform(ddf, input_col=cols)

    train, test = ddf.split(0.75)

    return train, test


def gaussian_classifier(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    from ddf_library.functions.ml.classification import GaussianNB
    nb = GaussianNB()\
        .fit(ddf_train, feature_col=cols, label_col='class')

    out_data = nb.transform(ddf_test, pred_col='prediction')
    out_data.show()
    print("ddf_train:", ddf_train.count_rows())
    print("ddf_test:", ddf_test.count_rows())
    return out_data


def knn_classifier(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    from ddf_library.functions.ml.classification import KNearestNeighbors
    knn = KNearestNeighbors(k=1)\
        .fit(ddf_train, feature_col=cols, label_col='class')

    out_data = knn.transform(ddf_test, pred_col='prediction')
    out_data.show()
    print("ddf_train:", ddf_train.count_rows())
    print("ddf_test:", ddf_test.count_rows())

    return out_data


def logistic_regression_classifier(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    from ddf_library.functions.ml.classification import LogisticRegression
    logr = LogisticRegression(max_iter=10, regularization=0.1, threshold=0.1)\
        .fit(ddf_train, feature_col=cols, label_col='class')

    out_data = logr.transform(ddf_test, pred_col='prediction')
    out_data.show()
    print("ddf_train:", ddf_train.count_rows())
    print("ddf_test:", ddf_test.count_rows())

    return out_data


def svm_classifier(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    from ddf_library.functions.ml.classification import SVM
    svm = SVM(max_iter=100, coef_lambda=1, coef_lr=0.01)\
        .fit(ddf_train, feature_col=cols, label_col='class',)

    out_data = svm.transform(ddf_test, pred_col='prediction')
    out_data.show()
    print("ddf_train:", ddf_train.count_rows())
    print("ddf_test:", ddf_test.count_rows())

    return out_data


def save_and_load(ddf_train, ddf_test):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    from ddf_library.functions.ml.classification import GaussianNB
    nb = GaussianNB() \
        .fit(ddf_train, feature_col=cols, label_col='class')
    path = 'hdfs://localhost:9000/saved-nb'
    nb.save_model(path, overwrite=True)
    del nb

    nb = GaussianNB().load_model(path)
    out_data = nb.transform(ddf_test, feature_col=cols, pred_col='prediction')
    out_data.show()
    print("ddf_train:", ddf_train.count_rows())
    print("ddf_test:", ddf_test.count_rows())
    return out_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="Testing Machine Learning Classifiers")
    parser.add_argument('-o', '--operation',
                        type=int,
                        required=True,
                        help="""
                             1. Gaussian Naive Bayes
                             2. kNN
                             3. Logistic Regressor
                             4. SVM
                             5. Save and load model
                            """)
    arg = vars(parser.parse_args())

    operation = arg['operation']
    list_operations = [gaussian_classifier, knn_classifier,
                       logistic_regression_classifier, svm_classifier,
                       save_and_load]
    cc = COMPSsContext()
    ddf1, ddf2 = base(cc)

    list_operations[operation - 1](ddf1, ddf2)
    cc.stop()
