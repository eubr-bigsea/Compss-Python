#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf_base import DDFSketch

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_delete_object, compss_wait_on

import numpy as np
import pandas as pd

__all__ = ['BinaryClassificationMetrics', 'MultilabelMetrics',
           'RegressionMetrics']


# TODO : replace ix to iat


class BinaryClassificationMetrics(DDFSketch):
    # noinspection PyUnresolvedReferences
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
    >>> print(bin_metrics.get_metrics())
    >>> # or using:
    >>> print(bin_metrics.confusion_matrix)
    >>> print(bin_metrics.accuracy)
    >>> print(bin_metrics.recall)
    >>> print(bin_metrics.precision)
    >>> print(bin_metrics.f1)
    """

    def __init__(self, label_col, pred_col, ddf_var, true_label=1):
        """
        :param label_col: Column name of true label values;
        :param pred_col: Column name of predicted label values;
        :param ddf_var: DDF;
        :param true_label: Value of True label (default is 1).
        """
        super(BinaryClassificationMetrics, self).__init__()

        df, nfrag, tmp = self._ddf_initial_setup(ddf_var)

        stage1 = [cme_stage1(df[f], label_col, pred_col) for f in range(nfrag)]
        merged_stage1 = merge_reduce(merge_stage1, stage1)
        compss_delete_object(stage1)
        
        result = cme_stage2_binary(merged_stage1, int(true_label))
        result = compss_wait_on(result)
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
    # noinspection PyUnresolvedReferences
    """
    Evaluator for multilabel classification.

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
    >>> print(metrics_multi.get_metrics())
    >>> # or using:
    >>> print(metrics_multi.confusion_matrix)
    >>> print(metrics_multi.precision_recall)
    >>> print(metrics_multi.accuracy)
    >>> print(metrics_multi.recall)
    >>> print(metrics_multi.precision)
    >>> print(metrics_multi.f1)
    """

    def __init__(self, label_col, pred_col, ddf_var):
        """
        :param label_col: Column name of true label values;
        :param pred_col: Column name of predicted label values;
        :param ddf_var: DDF.
        """

        super(MultilabelMetrics, self).__init__()

        df, nfrag, tmp = self._ddf_initial_setup(ddf_var)

        stage1 = [cme_stage1(df[f], label_col, pred_col) for f in range(nfrag)]
        merged_stage1 = merge_reduce(merge_stage1, stage1)
        compss_delete_object(stage1)

        result = cme_stage2(merged_stage1)
        compss_delete_object(merged_stage1)
        
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


@task(returns=1)
def cme_stage1(data, col_test, col_predicted):
    """Create a partial confusion matrix."""

    data = data.groupby([col_test,
                         col_predicted]).size().reset_index(name='counts')

    data[col_test] = data[col_test].astype(int)
    data[col_predicted] = data[col_predicted].astype(int)

    uniques = np.concatenate((data[col_test],
                              data[col_predicted]), axis=0)
    uniques = np.unique(uniques)

    df = pd.DataFrame(columns=uniques, index=uniques).fillna(0)
    del uniques

    # O(n_unique_labels)
    for real, pred, count in data[[col_test, col_predicted, 'counts']].values:
        df.at[real, pred] += count

    return df


@task(returns=1)
def merge_stage1(p1, p2):
    """Merge partial statistics."""
    p1 = p1.add(p2, fill_value=0)
    return p1


@task(returns=1)
def cme_stage2(confusion_matrix):
    """Generate the final evaluation."""
    confusion_matrix = confusion_matrix.fillna(0).astype(int)
    n_rows = confusion_matrix.sum().sum()
    labels = confusion_matrix.index
    accuracy = 0
    _precisions = []  # TPR
    _recalls = []  # FPR

    for i in labels:
        accuracy += confusion_matrix[i].at[i]

        true_positive = confusion_matrix[i].at[i]
        _precisions.append(true_positive / confusion_matrix.loc[i].sum())
        _recalls.append(true_positive / confusion_matrix[i].sum())

    accuracy = accuracy / n_rows

    _f1s = []
    for p, r in zip(_precisions, _recalls):
        _f1s.append(2 * (p * r) / (p + r))

    precision_recall = \
        pd.DataFrame(np.array([_precisions, _recalls, _f1s]).T,
                     columns=['Precision', 'Recall', "F-Measure"])

    precision = np.mean(_precisions)
    recall = np.mean(_recalls)
    f1 = 2 * (precision * recall) / (precision + recall)

    return [confusion_matrix, accuracy, precision,
            recall, f1, precision_recall]


# @local
@task(returns=1)
def cme_stage2_binary(confusion_matrix, true_label):
    """Generate the final evaluation (for binary classification)."""
    n_rows = confusion_matrix.sum().sum()
    labels = confusion_matrix.index
    accuracy = 0

    for i in labels:
        accuracy += confusion_matrix[i].at[i]

    true_positive = confusion_matrix[true_label].at[true_label]
    _precision = true_positive / confusion_matrix.loc[true_label].sum()
    _recall = true_positive / confusion_matrix[true_label].sum()
    accuracy = accuracy / n_rows
    _f1 = 2 * (_precision * _recall) / (_precision + _recall)

    return [confusion_matrix, accuracy, _precision, _recall, _f1]


class RegressionMetrics(DDFSketch):
    # noinspection PyUnresolvedReferences
    """
    RegressionModelEvaluation's methods.

    * **Mean Squared Error (MSE):** Is an estimator measures the average of the
      squares of the errors or deviations, that is, the difference between the
      estimator and what is estimated. MSE is a risk function, corresponding
      to the expected value of the squared error loss or quadratic loss. In
      other words, MSE tells you how close a regression line is to a set of
      points.

    * **Root Mean Squared Error (RMSE):** Is a frequently used measure of the
      differences between values (sample and population values) predicted by a
      model or an estimator and the values actually observed. The RMSE
      represents the sample standard deviation of the differences between
      predicted values and observed values.

    * **Mean Absolute Error (MAE):** Is a measure of difference between two
      continuous variables. Assume X and Y are variables of paired
      observations that express the same phenomenon. Is a quantity used to
      measure how close forecasts or predictions are to the eventual outcomes.

    * **Coefficient of Determination (R2):** Iis the proportion of the
      variance in the dependent variable that is predictable from the
      independent variable(s). Best possible score is 1.0 and it can be
      negative (because the model can be arbitrarily worse). A constant model
      that always predicts the expected value of y, disregarding the input
      features, would get a R^2 score of 0.0.

    * **Explained Variance:** Measures the proportion to which a mathematical
      model accounts for the variation (dispersion) of a given data set.

    :Example:

    >>> reg_metrics = RegressionMetrics(col_features=['x1', 'x2'],
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
        :param pred_col: Column name of predicted label values;
        :param data: DDF.
        """

        super(RegressionMetrics, self).__init__()

        df, nfrag, tmp = self._ddf_initial_setup(data)

        cols = [label_col, pred_col, col_features]
        partial = [rme_stage1(df[f], cols) for f in range(nfrag)]

        statistics = merge_reduce(rme_merge, partial)
        compss_delete_object(partial)
        
        result = rme_stage2(statistics)
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


@task(returns=1)
def rme_stage1(df, cols):
    """Generate the partial statistics of each fragment."""
    dim = 1
    col_test, col_predicted, col_features = cols
    sse_partial = ssy_partial = abs_error = sum_y = 0.0

    size = len(df)
    if size > 0:
        df.reset_index(drop=True, inplace=True)
        dim = len(col_features)

        sum_y = df[col_test].sum()
        ssy_partial = np.sum(df[col_test] ** 2)

        error = (df[col_test] - df[col_predicted]).values
        del df
        sse_partial = np.sum(error ** 2)
        abs_error = np.sum(np.absolute(error))

    table = np.array([size, sse_partial, ssy_partial, abs_error, sum_y])
    return [table, dim]


@task(returns=1)
def rme_merge(statistic1, statistic2):
    """Merge the partial statistics."""
    dim = max(statistic1[1], statistic2[1])
    statistic = statistic1[0] + statistic2[0]
    return [statistic, dim]


def rme_stage2(statistics):
    """Generate the final evaluation."""
    statistics = compss_wait_on(statistics)
    dim = statistics[1]
    n_rows, sse, ssy, abs_error, sum_y = statistics[0]

    if n_rows == 0:
        raise Exception("You must have inform a sample of rows.")

    y_mean = sum_y / n_rows
    ss0 = n_rows * (y_mean ** 2)

    # SST = Sum of Squares Total
    # SSE = Sum of Squared Errors
    # SSR = Regression Sum of Squares
    sst = ssy - ss0
    ssr = sst - sse

    r2 = ssr/sst

    # MSE = Mean Square Errors = Error Mean Square = Residual Mean Square
    den = n_rows - dim - 1

    if den == 0:
        msg = "You must have at least {} sample of rows.".format(dim)
        raise Exception(msg)

    mse = sse / den
    rmse = np.sqrt(mse)

    mae = abs_error / den

    # MSR = Mean Square of Regression
    msr = ssr/dim

    return r2, mse, rmse, mae, msr
