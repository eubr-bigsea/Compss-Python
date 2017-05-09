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

#-------------------------
#   Training
#
class SVM(object):

    def fit(self,train_data, settings, numFrag):

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

        numDim = len(train_data[0][0]) -1
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
    def updateWeight(self,coef_lr,grad,w):
        for i in xrange(len(w)):
            w[i] -=coef_lr*grad[i]
        return w

    @task(returns=list, isModifier = False)
    def calc_yp(self,train_data,w,numDim):
        ypp  = [0 for i in range(len(train_data))]

        for i in range(len(train_data)):
            ypp[i]=0
            for d in xrange(1,numDim):
                ypp[i]+=train_data[i][d]*w[d-1]

        return ypp

    @task(returns=list, isModifier = False)
    def calc_CostAndGrad(self,ypp,train_data,f,numDim,coef_lambda,w):
        cost  = [0,0]
        grad  = [0 for i in range(numDim)]

        for i in range(len(train_data)):
            if (train_data[i][0]*ypp[i]-1) < 0:
                cost[0]+=(1 - train_data[i][0]*ypp[i])

        for d in xrange(1,numDim):
            grad[d-1]=0
            if f is 0:
                grad[d-1]+=abs(coef_lambda * w[d-1])

            for i in range(len(train_data)):
                if (train_data[i][0]*ypp[i]-1) < 0:
                    grad[d-1] -= train_data[i][0] *train_data[i][d]

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


    def transform(self,test_data, w, numFrag):
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

        from pycompss.api.api import compss_wait_on
        result_p = [ self.predict_partial(test_data[f],w)  for f in range(numFrag) ]
        result   = [ mergeReduce(self.accumulate_prediction, result_p) ]
        result   =  compss_wait_on(result)

        return result[0]

    @task(returns=list, isModifier = False)
    def predict_partial(self,test_data,w):
        values = []
        for i in range(len(test_data)):
            values.append(self.predict_one(test_data[i],w))

        return values

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
