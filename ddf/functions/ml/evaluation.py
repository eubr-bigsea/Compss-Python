#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.local import *
from ddf.ddf import COMPSsContext, DDF, DDFSketch
import numpy as np
import pandas as pd

import sys
sys.path.append('../../')

__all__ = ['BinaryClassificationMetrics', 'MultilabelMetrics',
           'RegressionMetrics']


class BinaryClassificationMetrics(DDFSketch):
    """
    Evaluator for binary classification.

    * True Positive (TP) - label is positive and prediction is also positive
    * True Negative (TN) - label is negative and prediction is also negative
    * False Positive (FP) - label is negative but prediction is positive
    * False Negative (FN) - label is positive but prediction is negative

    Metrics:
        * Accuracy
        * Precision (Positive Predictive Value) = tp / (tp + fp)
        * Recall (True Positive Rate) = tp / (tp + fn)
        * F-measure = F1 = 2 * (precision * recall) / (precision + recall)
        * Confusion matrix

    :Example:

    >>> bin_metrics = BinaryClassificationMetrics(label_col='label',
    >>>                                           pred_col='pred', data=ddf1)
    >>> print bin_metrics.get_metrics()
    >>> # or using:
    >>> print bin_metrics.confusion_matrix
    >>> print bin_metrics.accuracy
    >>> print bin_metrics.recall
    >>> print bin_metrics.precision
    >>> print bin_metrics.f1
    """

    def __init__(self, label_col, pred_col, data, true_label=1):
        """
        :param label_col: Column name of true label values;
        :param pred_col: Colum name of predicted label values;
        :param data: DDF;
        :param true_label: Value of True label (default is 1).
        """
        super(BinaryClassificationMetrics, self).__init__()

        df, nfrag, tmp = self._ddf_inital_setup(data)

        stage1 = [CME_stage1(df[f], label_col, pred_col)
                  for f in range(nfrag)]
        merged_stage1 = merge_reduce(mergeStage1, stage1)

        result = CME_stage2_binary(merged_stage1, true_label)
        confusion_matrix, accuracy, precision, recall, f1 = result

        self.confusion_matrix = confusion_matrix
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.name = 'BinaryClassificationMetrics'

    def get_metrics(self):
        """
        :return: A pandas DataFrame with metrics.
        """
        table = pd.DataFrame([
            ["Accuracy", self.accuracy],
            ["Precision", self.precision],
            ["Recall", self.recall],
            ["F-measure (F1)", self.f1]
        ], columns=["Metric", "Value"])

        return table


class MultilabelMetrics(DDFSketch):
    """Evaluator for multilabel classification.

    * True Positive (TP) - label is positive and prediction is also positive
    * True Negative (TN) - label is negative and prediction is also negative
    * False Positive (FP) - label is negative but prediction is positive
    * False Negative (FN) - label is positive but prediction is negative

    Metrics:
        * Accuracy
        * Precision (Positive Predictive Value) = tp / (tp + fp)
        * Recall (True Positive Rate) = tp / (tp + fn)
        * F-measure = F1 = 2 * (precision * recall) / (precision + recall)
        * Confusion matrix
        * Precision_recall table

    :Example:

    >>> metrics_multi = MultilabelMetrics(label_col='label',
    >>>                                   pred_col='prediction', data=ddf1)
    >>> print metrics_multi.get_metrics()
    >>> # or using:
    >>> print metrics_multi.confusion_matrix
    >>> print metrics_multi.precision_recall
    >>> print metrics_multi.accuracy
    >>> print metrics_multi.recall
    >>> print metrics_multi.precision
    >>> print metrics_multi.f1
    """

    def __init__(self, label_col, pred_col, data):
        """
        :param label_col: Column name of true label values;
        :param pred_col: Colum name of predicted label values;
        :param data: DDF.
        """

        super(MultilabelMetrics, self).__init__()

        df, nfrag, tmp = self._ddf_inital_setup(data)

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
        """
        :return: A pandas DataFrame with metrics.
        """
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


@local
def CME_stage2_binary(confusion_matrix, true_label):
    """Generate the final evaluation (for binary classification)."""
    total_size = confusion_matrix.sum().sum()
    labels = confusion_matrix.index
    acertos = 0

    for i in labels:
        acertos += confusion_matrix[i].ix[i]

    TP = confusion_matrix[true_label].ix[true_label]
    Precision = float(TP) / confusion_matrix.ix[true_label].sum()
    Recall = float(TP) / confusion_matrix[true_label].sum()
    Accuracy = float(acertos) / total_size
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    return [confusion_matrix, Accuracy, Precision, Recall, F1]


class RegressionMetrics(DDFSketch):
    """RegressionModelEvaluation's methods.

    * **Mean Squared Error (MSE):** Is an estimator measures the average of the
      squares of the errors or deviations, that is, the difference between the
      estimator and what is estimated. MSE is a risk function, corresponding
      to the expected value of the squared error loss or quadratic loss. In
      other words, MSE tells you how close a regression line is to a set of
      points.

    * **Root Mean Squared Error (RMSE):** Is a frequently used measure of the
      differences between values (sample and population values) predicted by a
      model or an estimator and the values actually observed. The RMSD
      represents the sample standard deviation of the differences between
      predicted values and observed values.

    * **Mean Absolute Error (MAE):** Is a measure of difference between two
      continuous variables. Assume X and Y are variables of paired
      observations that express the same phenomenon. Is a quantity used to
      measure how close forecasts or predictions are to the eventual outcomes.

    * **Coefficient of Determination (R2):** Iis the proportion of the
      variance in the dependent variable that is predictable from the
      independent variable(s).

    * **Explained Variance:** Measures the proportion to which a mathematical
      model accounts for the variation (dispersion) of a given data set.

    :Example:

    >>> reg_metrics = RegressionMetrics(col_features='features',
    >>>                                 label_col='label', pred_col='pred',
    >>>                                 data=data)
    >>> print reg_metrics.get_metrics()
    >>> # or using:
    >>> print reg_metrics.r2
    >>> print reg_metrics.mse
    >>> print reg_metrics.rmse
    >>> print reg_metrics.mae
    >>> print reg_metrics.msr
    """

    def __init__(self, col_features, label_col, pred_col, data):
        """
        :param col_features: Column name of features values;
        :param label_col: Column name of true label values;
        :param pred_col: Colum name of predicted label values;
        :param data: DDF.
        """

        super(RegressionMetrics, self).__init__()

        df, nfrag, tmp = self._ddf_inital_setup(data)

        cols = [label_col, pred_col, col_features]
        partial = [RME_stage1(df[f], cols) for f in range(nfrag)]
        statistics = merge_reduce(mergeRME, partial)
        result = RME_stage2(statistics)

        r2, mse, rmse, mae, msr = result

        self.r2 = r2
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.msr = msr
        self.name = 'RegressionMetrics'

    def get_metrics(self):
        """
        :return: A pandas DataFrame with metrics.
        """

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
    sse_partial = ssy_partial = abs_error = sum_y = 0.0

    if len(df) > 0:
        df.reset_index(drop=True, inplace=True)
        head = df.loc[0, col_features]
        if isinstance(head, list):
            dim = len(head)

        error = (df[col_test] - df[col_predicted]).values

        sse_partial = np.sum(error**2)
        abs_error = np.sum(np.absolute(error))

        sum_y = df[col_test].sum()
        ssy_partial = np.sum(df[col_test]**2)

    size = len(df)
    table = np.array([size, sse_partial, ssy_partial, abs_error, sum_y])
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

    y_mean = float(sum_y) / N
    SS0 = N*np.square(y_mean)

    SST = SSY - SS0  # SST is the total sum of squares
    SSR = float(SST - SSE)

    R2 = SSR/SST

    # MSE = Mean Square Errors = Error Mean Square = Residual Mean Square
    MSE = float(SSE)/(N-dim-1)
    RMSE = np.sqrt(MSE)

    MAE = float(abs_error)/(N-dim-1)

    # MSR = MSRegression = Mean Square of Regression
    MSR = SSR/dim

    return R2, MSE, RMSE, MAE, MSR

