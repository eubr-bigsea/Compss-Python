#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# Developed by Lucas Miguel Ponce
# Mestrando em Ciências da Computação - UFMG
# <lucasmsp@gmail.com>
#

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks
from pycompss.api.api import compss_wait_on
import pandas as pd
import numpy as np

#-------------------------
#   Training
#
class SVM(object):
    #OBS.: Senao for usar o COST, tem como otimizar o codigo
    def fit(self,data, settings, numFrag):

        """
            SVM is a supervised learning model used for binary classification.
            Given a set of training examples, each marked as belonging to one or
            the other of two categories, an SVM training algorithm builds a model
            that assigns new examples to one category or the other, making it a
            non-probabilistic binary linear classifier.

            An SVM model is a representation of the examples as points in space,
            mapped so that the examples of the separate categories are divided by
            a clear gap that is as wide as possible. New examples are then mapped
            into that same space and predicted to belong to a category based on
            which side of the gap they fall.

            The algorithm reads a dataset composed by labels (-1.0 or 1.0) and
            features (numeric fields).

            :param train_data:  The data (splitted) to train the model
            :param settings:  A dictionary with some necessary parameters:
                                - coef_lambda: Regularization parameter (float)
                                - coef_lr: Learning rate parameter (float)
                                - coef_threshold: Tolerance for stopping criterion (float)
                                - coef_maxIters: Number max of iterations (integer)
            :param numFrag:       Num of fragments

            :return A model (a np.array)
        """


        coef_lambda     = float(settings['coef_lambda'])
        coef_lr         = float(settings['coef_lr'])
        coef_threshold  = float(settings['coef_threshold'])
        coef_maxIters   =   int(settings['coef_maxIters'])

        label    = settings['label']
        features = settings['features']
        w = [0 for i in range(1)] #initial

        for it in range(coef_maxIters):
            cost_grad_p = [self.calc_CostAndGrad(data[f], f, coef_lambda,
                                                 w,label,features)
                                                 for f in range(numFrag)]
            cost_grad   =  mergeReduce(self.accumulate_CostAndGrad, cost_grad_p)
            #cost_grad   =  compss_wait_on(cost_grad)

            #grad = cost_grad[0][1]
            #cost = cost_grad[0][0][0]
            #print "[INFO] - Current Cost %.4f" % (cost)
            #if cost < coef_threshold:
            #    print "[INFO] - Cost %.4f" % (cost)
            #    break

            w = self.updateWeight(coef_lr,cost_grad,w)

        return w

    @task(returns=list, isModifier = False)
    def updateWeight(self,coef_lr,grad,w):
        dim = len(grad[1])
        if(dim!=len(w)):
            w = [0 for i in range(dim)]

        for i in xrange(len(w)):
            w[i] -=coef_lr*grad[1][i]
        return w

    def get_dimension(self,row):
        if isinstance(row, list):
            numDim = len(row)
        else:
            numDim = 1
        return numDim


    @task(returns=list, isModifier = False)
    def calc_CostAndGrad(self,train_data,f,coef_lambda,w,label,features):
        numDim = self.get_dimension(data.iloc[0][X])

        ypp   = [0 for i in range(len(train_data))]
        cost  = [0,0]
        grad  = [0 for i in range(numDim)]

        if numDim != len(w):
            w = [0 for i in range(numDim)] #initial

        if (len(train_data)):
            for i in range(len(train_data)):
                ypp[i]=0
                for d in xrange(0,numDim):
                    ypp[i]+=train_data.iloc[i][features][d]*w[d]

                if (train_data.iloc[i][label] * ypp[i] -1) < 0:
                    cost[0]+=(1 - train_data.iloc[i][label] * ypp[i])


            for d in range(numDim):
                grad[d]=0
                if f is 0:
                    grad[d]+=abs(coef_lambda * w[d])

                for i in range(len(train_data)):
                    if (train_data.iloc[i][label]*ypp[i]-1) < 0:
                        grad[d] -= train_data.iloc[i][label]*train_data.iloc[i][features][d]

        return [cost,grad]

    @task(returns=list, isModifier = False)
    def accumulate_CostAndGrad(self,cost_grad_p1,cost_grad_p2):
        cost_p1 = cost_grad_p1[0]
        cost_p2 = cost_grad_p2[0]
        for i in range(len(cost_p1)):
            cost_p1[i]+=cost_p2[i]

        grad_p1 = cost_grad_p1[1]
        grad_p2 = cost_grad_p2[1]
        for d in range(len(grad_p1)):
            grad_p1[d]+=grad_p2[d]

        return [cost_p1, grad_p1]

    #------------------------------------------------------------------------

    def transform(self,data, settings, numFrag):
        """
            SVM is a supervised learning model used for binary classification.
            Given a set of training examples, each marked as belonging to one or
            the other of two categories, an SVM training algorithm builds a model
            that assigns new examples to one category or the other, making it a
            non-probabilistic binary linear classifier.

            An SVM model is a representation of the examples as points in space,
            mapped so that the examples of the separate categories are divided by
            a clear gap that is as wide as possible. New examples are then mapped
            into that same space and predicted to belong to a category based on
            which side of the gap they fall.

            The algorithm reads a dataset composed by labels (-1.0 or 1.0) and
            features (numeric fields).

            :param test_data: The list (splitted) to predict.
            :param w: A model already trained (np.array)
            :param numFrag:       Num of fragments, if -1 data is considered chunked

            :return: A list with the labels
        """

        w = settings['model']
        features = settings['features']
        predictedLabel = settings['new_name'] if 'new_name' in settings else "{}_predited".format(label)

        result_p   = [ self.predict_partial(data[f],w,predictedLabel,features)  for f in range(numFrag) ]

        return result_p


    @task(returns=list, isModifier = False)
    def predict_partial(self,data,w,predictedLabel,features):

        if len(data)>0:
            values = [0 for i in range(len(data))]
            for i in range(len(data)):
                values[i] = self.predict_one(data.iloc[i][features],w)

            data[predictedLabel] =  pd.Series(values).values

            return data
        else:
            return []

    def predict_one(self,test_xi, w):
        pre = 0
        for i in range(len(test_xi)):

            pre+=test_xi[i]*w[i]
        if pre >= 0:
            return 1.0
        return -1.0



    @task(returns=list, isModifier = False)
    def accumulate_prediction(self,result_p1,result_p2):
        return result_p1+result_p2
