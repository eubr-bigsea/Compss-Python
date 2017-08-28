#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.api.api import compss_wait_on

import numpy as np
import pandas as pd
import sys



def RegressionModelEvaluation(data,settings,numFrag):
    """
        Metric	Definition
        * Mean Squared Error (MSE): Is an estimator measures the average of the
        squares of the errors or deviations—that is, the difference between the
        estimator and what is estimated. MSE is a risk function, corresponding
        to the expected value of the squared error loss or quadratic loss. In
        other words, MSE tells you how close a regression line is to a set of
        points.

        * Root Mean Squared Error (RMSE): Is a frequently used measure of the
        differences between values (sample and population values) predicted by a
        model or an estimator and the values actually observed. The RMSD represents
        the sample standard deviation of the differences between predicted values
        and observed values.

        * Mean Absolute Error (MAE): Is a measure of difference between two
        continuous variables. Assume X and Y are variables of paired observations
        that express the same phenomenon. Is a quantity used to measure how close
        forecasts or predictions are to the eventual outcomes.

        * Coefficient of Determination (R2): Iis the proportion of the variance in
        the dependent variable that is predictable from the independent variable(s).

        * Explained Variance: Measures the proportion to which a mathematical model
        accounts for the variation (dispersion) of a given data set.
    """
    dim = settings['dimension'] if "dimension" in settings else 1

    if settings['metric'] in ['MSE','RMSE']:
        col_predicted = settings['pred_col']
        col_test      = settings['test_col']
        partial = [RME_stage1(data[f],col_test,col_predicted) for f in range(numFrag)]
        statistics = mergeReduce(mergeRME,partial)
        result = RME_stage2(statistics, dim)

    return result

@task(returns=list)
def RME_stage1(df,col_test,col_predicted):
    error = (df[col_test] - df[col_predicted]).values
    SSE_partial = np.sum(np.square(error))
    abs_error = np.sum(np.absolute(error))
    sum_y = 0
    SSY_partial = 0
    for y in df[col_test].values:
        sum_y+=y
        SSY_partial+=np.square(y)

    N = len(df)
    return np.array([N, SSE_partial, SSY_partial, abs_error, sum_y ]).astype(float)

@task(returns=list)
def mergeRME(pstatistic1,pstatistic2):
    pstatistic = pstatistic1 + pstatistic2
    return pstatistic

@task(returns=list)
def RME_stage2(statistics,dim):
    N, SSE, SSY, abs_error, sum_y = statistics
    y_mean = sum_y / N
    SS0 = N*(y_mean)**2

    SST = SSY - SS0
    SSR = SST - SSE
    R2 = SSR/SST

    MSE = SSE/(N-dim-1)
    RMSE = np.sqrt(MSE)
    MAE = abs_error

    Explained_Variance = 1 - SSE/SST

    result =  pd.DataFrame([
                            ["R^2",R2],
                            ["Mean Squared Error (MSE)",MSE],
                            ["Root Mean Squared Error (RMSE)",RMSE],
                            ["Mean Absolute Error (MAE)",MAE],
                            ["Explained Variance", Explained_Variance]
                        ],columns=["Metric","Value"])
    return result
