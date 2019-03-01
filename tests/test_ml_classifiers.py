#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF
import numpy as np


def ml_classifiers_part1():

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
                  'Iris-virginica': 1.0}, subset=['class'])

    # assembling a group of attributes as features and removing them after
    from ddf.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["sepal_length", "sepal_width",
                                           "petal_length", "petal_width"],
                                output_col="features")
    ddf = assembler.transform(ddf).drop(["sepal_length", "sepal_width",
                                         "petal_length", "petal_width"])

    # scaling using StandardScaler
    from ddf.functions.ml.feature import StandardScaler
    ddf = StandardScaler(input_col='features', output_col='features')\
        .fit_transform(ddf)

    # splitting 25% to use as training set and 75% as test
    ddf_train, ddf_test = ddf.split(0.25)

    from ddf.functions.ml.classification import GaussianNB
    nb = GaussianNB(feature_col='features', label_col='class')\
        .fit(ddf_train)
    nb.save_model('/gaussian_nb')  # save this fitted model in HDFS
    ddf_test = nb.transform(ddf_test)

    from ddf.functions.ml.classification import KNearestNeighbors
    knn = KNearestNeighbors(k=1, feature_col='features', label_col='class')\
        .fit(ddf_train)
    knn.save_model('/knn')
    ddf_test = knn.transform(ddf_test)

    from ddf.functions.ml.classification import LogisticRegression
    logr = LogisticRegression(feature_col='features', label_col='class',
                              max_iters=10).fit(ddf_train)
    logr.save_model('/logistic_regression')
    f = lambda row: -1.0 if row['prediction_LogReg'] == 0.0 else 1.0
    ddf_test = logr.transform(ddf_test).map(f, 'prediction_LogReg')

    from ddf.functions.ml.classification import SVM
    svm = SVM(feature_col='features', label_col='class',
              max_iters=10).fit(ddf_train)
    svm.save_model('/svm')
    ddf_test = svm.transform(ddf_test)
    #
    from ddf.functions.ml.evaluation import MultilabelMetrics, \
        BinaryClassificationMetrics

    metrics_bin = BinaryClassificationMetrics(label_col='class',
                                              true_label=1.0,
                                              pred_col='prediction_LogReg',
                                              data=ddf_test)

    metrics_multi = MultilabelMetrics(label_col='class',
                                      pred_col='prediction_GaussianNB',
                                      data=ddf_test)

    # Retrieve the dataset from PyCOMPSs Future objects as a single DataFrame
    print ddf_test.show(20)

    # Get some metrics
    print "Metrics for Logistic Regression"
    print metrics_bin.get_metrics()

    print "Metrics for Gaussian Naive Bayes"
    print metrics_multi.get_metrics()
    print metrics_multi.confusion_matrix
    print metrics_multi.precision_recall


def ml_classifiers_part2():

    # loading a csv file from HDFS
    ddf = DDF().load_text('/iris-dataset.csv', num_of_parts=4, sep=',',
                          dtype={'class': np.dtype('O'),  # string
                                 'sepal_length': np.float64,
                                 'sepal_width': np.float64,
                                 'petal_length': np.float64,
                                 'petal_width': np.float64
                                 }) \
        .replace({'Iris-setosa': 1.0,
                  'Iris-versicolor': -1.0,
                  'Iris-virginica': 1.0}, subset=['class'])

    # assembling a group of attributes as features and removing them after
    from ddf.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=["sepal_length", "sepal_width",
                                           "petal_length", "petal_width"],
                                output_col="features")
    ddf = assembler.transform(ddf).drop(["sepal_length", "sepal_width",
                                         "petal_length", "petal_width"])

    # scaling using StandardScaler
    from ddf.functions.ml.feature import StandardScaler
    scaler = StandardScaler(input_col='features', output_col='features',
                            with_mean=True, with_std=True).fit(ddf)
    ddf = scaler.transform(ddf)

    # Loading previous fitted ml models
    from ddf.functions.ml.classification import GaussianNB,\
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
    df = nb.transform(ddf).show()

    print 'All classifiers:\n', df


if __name__ == '__main__':
    print "_____Testing Machine Learning Classifiers_____"

    ml_classifiers_part1()
    # ml_classifiers_part2()
