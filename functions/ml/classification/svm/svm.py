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

import numpy as np
import pandas as pd
#-------------------------
#   Training
#
class SVM(object):

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

        columns = settings['labels']+settings['features']
        train_data = [self.format_data(data[i],columns) for i in range(numFrag)]

        numDim = len(settings['features'])
        w = [0 for i in range(numDim)]

        for it in range(coef_maxIters):
            from pycompss.api.api import compss_wait_on
            yp          = [ self.calc_yp(train_data[f],w,numDim)  for f in range(numFrag) ]
            cost_grad_p = [ self.calc_CostAndGrad(yp[f],train_data[f],f,numDim,coef_lambda,w)  for f in range(numFrag) ]
            cost_grad   = [ mergeReduce(self.accumulate_CostAndGrad, cost_grad_p) ]
            cost_grad   =  compss_wait_on(cost_grad)

            grad = cost_grad[0][1]
            cost = cost_grad[0][0][0]
            print "[INFO] - Current Cost %.4f" % (cost)
            if cost < coef_threshold:
                print "[INFO] - Cost %.4f" % (cost)
                break

            w = self.updateWeight(coef_lr,grad,w)

        return w


    @task(returns=list, isModifier = False)
    def format_data(self,data,columns):
        train_data = []
        tmp = np.array(data[columns].values)

        for j in range(len(tmp)):
            train_data.append([tmp[j][0],tmp[j][1:,]])


        return train_data

    @task(returns=list, isModifier = False)
    def updateWeight(self,coef_lr,grad,w):
        for i in xrange(len(w)):
            w[i] -=coef_lr*grad[i]
        return w

    @task(returns=list, isModifier = False)
    def calc_yp(self,train_data,w,numDim):
        ypp  = [0 for i in range(len(train_data))]

        for i in range(len(train_data)):
            print train_data[i]
            ypp[i]=0
            for d in xrange(0,numDim):
                ypp[i]+=train_data[i][1][d]*w[d]

        print ypp
        return ypp

    @task(returns=list, isModifier = False)
    def calc_CostAndGrad(self,ypp,train_data,f,numDim,coef_lambda,w):
        cost  = [0,0]
        grad  = [0 for i in range(numDim)]

        if (len(train_data)):
            for i in range(len(train_data)):
                print "---"
                print train_data[i][0]
                print ypp[i]
                print "---"
                if (train_data[i][0] * ypp[i] -1) < 0:
                    cost[0]+=(1 - train_data[i][0]*ypp[i])


            for d in range(numDim):
                grad[d]=0
                if f is 0:
                    grad[d]+=abs(coef_lambda * w[d])

                for i in range(len(train_data)):
                    if (train_data[i][0]*ypp[i]-1) < 0:
                        grad[d] -= train_data[i][0]*train_data[i][1][d]

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
    #   Testing
    #


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

        error =0
        values = []
        w = settings['model']
        features = settings['features']
        label = "_".join(i for i in settings['labels'])
        #print label

        test_data = [[] for i in range(numFrag)]
        for i in range(numFrag):
            test_data[i] = np.array(data[i][features].values)

        #print test_data
        from pycompss.api.api import compss_wait_on
        result_p = [ self.predict_partial(test_data[f],w,data[f],label)  for f in range(numFrag) ]
        #result   = [ mergeReduce(self.accumulate_prediction, result_p) ]
        #result   =  compss_wait_on(result)
        #print result
        return result_p

    @task(returns=list, isModifier = False)
    def predict_partial(self,test_data,w,data,label):
        values = [0 for i in range(len(test_data))]

        new_column = label +"_predited"
        if len(test_data)>0:
            for i in range(len(test_data)):
                values[i] = self.predict_one(test_data[i],w)

            data[new_column] =  pd.Series(values).values
            print data
            return data
        else:
            return []

    def predict_one(self,test_xi, w):
        pre = 0
        print test_xi
        print w
        for i in range(len(test_xi)):

            pre+=test_xi[i]*w[i]
        if pre >= 0:
            return 1.0
        return -1.0



    @task(returns=list, isModifier = False)
    def accumulate_prediction(self,result_p1,result_p2):
        return result_p1+result_p2
