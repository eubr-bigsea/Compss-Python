#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on
from pycompss.api.local import *
from ddf.ddf import COMPSsContext, DDF, ModelDDS
import numpy as np
import pandas as pd
import uuid

import sys
sys.path.append('../../')

__all__ = ['BinaryClassificationMetrics', 'MultilabelMetrics',
           'RegressionMetrics']


class BinaryClassificationMetrics(object):
    """

    areaUnderPR: Computes the area under the precision-recall curve.
    areaUnderROC: Computes the area under the receiver operating characteristic (ROC) curve.
    ["Accuracy", Accuracy],
    ["Precision", Precision],
    ["Recall", Recall],
    ["F-measure (F1)", F1]
    """

    def __init__(self, label_col, pred_col, data, true_label=1):

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        stage1 = [CME_stage1(df[f], label_col, pred_col)
                  for f in range(nfrag)]
        merged_stage1 = merge_reduce(mergeStage1, stage1)

        result = CME_stage2_binary(merged_stage1, true_label)
        result = compss_wait_on(result)
        confusion_matrix, accuracy, precision, recall, f1 = result

        self.confusion_matrix = confusion_matrix
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.name = 'BinaryClassificationMetrics'

    def get_metrics(self):
        table = pd.DataFrame([
            ["Accuracy", self.accuracy],
            ["Precision", self.precision],
            ["Recall", self.recall],
            ["F-measure (F1)", self.f1]
        ], columns=["Metric", "Value"])

        return table


class MultilabelMetrics(object):
    """ClassificationModelEvaluation.

    * True Positive (TP) - label is positive and prediction is also positive
    * True Negative (TN) - label is negative and prediction is also negative
    * False Positive (FP) - label is negative but prediction is positive
    * False Negative (FN) - label is positive but prediction is negative

    Metrics:
        * Accuracy
        * Precision (Positive Predictive Value) = tp / (tp + fp)
        * Recall (True Positive Rate) = tp / (tp + fn)
        * F-measure = F1 = 2 * (precision * recall) / (precision + recall)
    """

    def __init__(self, label_col, pred_col, data):

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        stage1 = [CME_stage1(df[f], label_col, pred_col)
                  for f in range(nfrag)]
        merged_stage1 = merge_reduce(mergeStage1, stage1)

        result = CME_stage2(merged_stage1)
        result = compss_wait_on(result)
        confusion_matrix, accuracy, precision, recall, f1, precision_recall \
            = result

        self.confusion_matrix = confusion_matrix
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.precision_recall = precision_recall
        self.name = 'MultilabelMetrics'

    def get_metrics(self):
        table = pd.DataFrame([
            ["Accuracy", self.accuracy],
            ["Precision", self.precision],
            ["Recall", self.recall],
            ["F-measure (F1)", self.f1]
        ], columns=["Metric", "Value"])

        return table


@task(returns=list)
def CME_stage1(data, col_test, col_predicted):
    """Create a partial confusion matrix."""
    Reals = data[col_test].values
    Preds = data[col_predicted].values

    labels = np.unique(
                np.concatenate(
                    (data[col_test].unique(), data[col_predicted].unique()),
                    0))

    df = pd.DataFrame(columns=labels, index=labels).fillna(0)

    for real, pred in zip(Reals, Preds):
        df.loc[real, pred] += 1

    return df


@task(returns=list)
def mergeStage1(p1, p2):
    """Merge partial statistics."""
    p1 = p1.add(p2, fill_value=0)
    return p1


@task(returns=list)
def CME_stage2(confusion_matrix):
    """Generate the final evaluation."""
    N = confusion_matrix.sum().sum()
    labels = confusion_matrix.index
    acertos = 0
    Precisions = []  # TPR
    Recalls = []  # FPR
    for i in labels:
        acertos += confusion_matrix[i].ix[i]
        TP = confusion_matrix[i].ix[i]
        Precisions.append(float(TP) / confusion_matrix.ix[i].sum())
        Recalls.append(float(TP) / confusion_matrix[i].sum())

    Accuracy = float(acertos) / N

    F1s = []
    for p, r in zip(Precisions, Recalls):
        F1s.append(2 * (p * r) / (p + r))

    precision_recall = \
        pd.DataFrame(np.array([Precisions, Recalls, F1s]).T,
                     columns=['Precision', 'Recall', "F-Mesure"])

    Precision = np.mean(Precisions)
    Recall = np.mean(Recalls)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    return [confusion_matrix, Accuracy, Precision, Recall, F1, precision_recall]


@task(returns=list)
def CME_stage2_binary(confusion_matrix, true_label):
    """Generate the final evaluation (for binary classification)."""
    N = confusion_matrix.sum().sum()
    labels = confusion_matrix.index
    acertos = 0

    for i in labels:
        acertos += confusion_matrix[i].ix[i]

    TP = confusion_matrix[true_label].ix[true_label]
    Precision = float(TP) / confusion_matrix.ix[true_label].sum()
    Recall = float(TP) / confusion_matrix[true_label].sum()
    Accuracy = float(acertos) / N
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    return [confusion_matrix, Accuracy, Precision, Recall, F1]


class RegressionMetrics(object):
    """RegressionModelEvaluation's methods.

    * Mean Squared Error (MSE): Is an estimator measures the average of the
    squares of the errors or deviations—that is, the difference between the
    estimator and what is estimated. MSE is a risk function, corresponding
    to the expected value of the squared error loss or quadratic loss. In
    other words, MSE tells you how close a regression line is to a set of
    points.

    * Root Mean Squared Error (RMSE): Is a frequently used measure of the
    differences between values (sample and population values) predicted by a
    model or an estimator and the values actually observed. The RMSD
    represents the sample standard deviation of the differences between
    predicted values and observed values.

    * Mean Absolute Error (MAE): Is a measure of difference between two
    continuous variables. Assume X and Y are variables of paired
    observations
    that express the same phenomenon. Is a quantity used to measure how
    close
    forecasts or predictions are to the eventual outcomes.

    * Coefficient of Determination (R2): Iis the proportion of the
    variance in
    the dependent variable that is predictable from the independent
    variable(s).

    * Explained Variance: Measures the proportion to which a mathematical
    model
    accounts for the variation (dispersion) of a given data set.
    """

    def __init__(self, col_features, label_col, pred_col, data):

        tmp = data.cache()
        df = COMPSsContext.tasks_map[tmp.last_uuid]['function'][0]
        nfrag = len(df)

        cols = [label_col, pred_col, col_features]
        partial = [RME_stage1(df[f], cols) for f in range(nfrag)]
        statistics = merge_reduce(mergeRME, partial)

        # partial_ssy = [RME_stage2(df[f], cols, statistics)
        #                for f in range(nfrag)]
        # ssy = merge_reduce(merge_stage2, partial_ssy)


        result = RME_stage2(statistics)

        r2, mse, rmse, mae, msr = result

        self.r2 = r2
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.msr = msr
        self.name = 'RegressionMetrics'

    def get_metrics(self):

        result = pd.DataFrame([
            ["R^2 (Explained Variance)", self.r2],
            ["Mean Squared Error (MSE)", self.mse],
            ["Root Mean Squared Error (RMSE)", self.rmse],
            ["Mean Absolute Error (MAE)", self.mae],
            ['Mean Square of Regression (MSR)', self.msr]
        ], columns=["Metric", "Value"])
        return result


@task(returns=list)
def RME_stage1(df, cols):
    """Generate the partial statistics of each fragment."""
    dim = 1
    col_test, col_predicted, col_features = cols
    SSE_partial = SSY_partial = abs_error = sum_y = 0

    if len(df) > 0:
        df.reset_index(drop=True, inplace=True)
        head = df.loc[0, col_features]
        if isinstance(head, list):
            dim = len(head)

        error = (df[col_test] - df[col_predicted]).values
        SSE_partial = np.sum(np.square(error))
        abs_error = np.sum(np.absolute(error))

        for y, yi in zip(df[col_test].values, df[col_predicted].values):
            sum_y += y
            SSY_partial += np.square(yi)

    size = len(df)
    table = np.array([size, SSE_partial, SSY_partial, abs_error, sum_y])
    table = table.astype(float)
    return [table, dim]


@task(returns=list)
def mergeRME(pstatistic1, pstatistic2):
    """Merge the partial statistics."""
    dim = max(pstatistic1[1], pstatistic2[1])
    pstatistic = pstatistic1[0] + pstatistic2[0]
    return [pstatistic, dim]



@local
def RME_stage2(statistics):
    """Generate the final evaluation."""
    dim = statistics[1]
    N, SSE, SSY, abs_error, sum_y = statistics[0]
    y_mean = sum_y / N
    SS0 = N*(y_mean)**2

    SST = SSY - SS0
    SSR = SST - SSE
    R2 = SSR/SST

    # MSE = Mean Square Errors = Error Mean Square = Residual Mean Square
    MSE = SSE/(N-dim-1)
    RMSE = np.sqrt(MSE)

    MAE = abs_error/(N-dim-1)

    # MSR = MSRegression = Mean Square of Regression
    MSR = SSR/dim

    return R2, MSE, RMSE, MAE, MSR

