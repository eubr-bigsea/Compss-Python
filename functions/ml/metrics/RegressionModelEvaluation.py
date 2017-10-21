#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.parameter     import *
from pycompss.api.task          import task
from pycompss.functions.reduce  import mergeReduce

import numpy as np
import pandas as pd

class RegressionModelEvaluation(object):

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
        model or an estimator and the values actually observed. The RMSD
        represents the sample standard deviation of the differences between
        predicted values and observed values.

        * Mean Absolute Error (MAE): Is a measure of difference between two
        continuous variables. Assume X and Y are variables of paired observations
        that express the same phenomenon. Is a quantity used to measure how close
        forecasts or predictions are to the eventual outcomes.

        * Coefficient of Determination (R2): Iis the proportion of the variance in
        the dependent variable that is predictable from the independent variable(s).

        * Explained Variance: Measures the proportion to which a mathematical model
        accounts for the variation (dispersion) of a given data set.
    """

    def calculate(self,data,settings,numFrag):
        """
        calculate():

        :param data:        A list with pandas dataframe with, at least,
                            two columns, one with the predicted label
                            (returned by the classificator) and other with
                            the true value of each record.
        :param settings:    A dictionary that contains:
            - test_col:     The field with the true label/value;
            - pred_col:     The field with the predicted label/value;
            - features:     The field with the features;
        :param numFrag:     A number of fragments;
        :return             A dataframe with the metrics
        """

        if any(['test_col' not in settings,
                'pred_col'  not in settings,
                'features' not in settings]):
           raise Exception( "You must inform the `test_col`, "
                            "`pred_col` and `features` fields.")

        col_predicted = settings['pred_col']
        col_test      = settings['test_col']
        col_features  = settings['features']
        partial = [self.RME_stage1(data[f],col_test,col_predicted,
                    col_features) for f in range(numFrag)]
        statistics = mergeReduce(self.mergeRME,partial)
        result = self.RME_stage2(statistics)

        return result

    @task(returns=list, isModifier = False)
    def RME_stage1(self,df,col_test,col_predicted,col_features):

        dim = 1
        SSE_partial = SSY_partial = abs_error = sum_y = 0
        if len(df)>0:
            df.reset_index(drop=True, inplace=True)
            head = df.loc[0,col_features]
            if isinstance(head,list):
                dim = len(head)

            error = (df[col_predicted]-df[col_test]).values
            SSE_partial = np.sum(np.square(error))
            abs_error = np.sum(np.absolute(error))

            for y in df[col_test].values:
                sum_y+=y
                SSY_partial+=np.square(y)

        N = len(df)
        table = np.array([N, SSE_partial, SSY_partial, abs_error, sum_y ])
        table = table.astype(float)
        return [table, dim]

    @task(returns=list, isModifier = False)
    def mergeRME(self,pstatistic1,pstatistic2):
        dim = max(pstatistic1[1], pstatistic2[1])
        pstatistic = pstatistic1[0] + pstatistic2[0]
        return [pstatistic,dim]

    @task(returns=list, isModifier = False)  #@local
    def RME_stage2(self,statistics):

        dim = statistics[1]
        N, SSE, SSY, abs_error, sum_y = statistics[0]
        y_mean = sum_y / N
        SS0 = N*(y_mean)**2

        SST = SSY - SS0
        #SSR = SST - SSE
        R2 = 1 - SSE/SST # or SSR/SST

        #MSE = Mean Square Errors = Error Mean Square = Residual Mean Square
        MSE = SSE/(N-dim-1)
        RMSE = np.sqrt(MSE)

        MAE = abs_error/(N-dim-1)

        #MSR = MSRegression = Mean Square of Regression
        MSR =  SSE/dim

        result =  pd.DataFrame([
                                ["R^2 (Explained Variance)",R2],
                                ["Mean Squared Error (MSE)",MSE],
                                ["Root Mean Squared Error (RMSE)",RMSE],
                                ["Mean Absolute Error (MAE)",MAE],
                                ['Mean Square of Regression (MSR)',MSR]
                            ],columns=["Metric","Value"])
        return result
