#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.functions.data    import chunks
from pycompss.api.api import compss_wait_on
import math
import numpy as np

class logisticRegression(object):



    def fit(self,data,settings,numFrag):

        features = settings['features']
        label    = settings['label']
        alpha = settings['alpha']
        iters = settings['iters']
        threshold = settings['threshold']
        reg = settings['regularization']
        parameters = self.ComputeCoeffs(data, features, label, alpha,
                                                iters, threshold, reg, numFrag)

        return parameters

    def sigmoid(self ,x, w):
        """
        Evaluate the sigmoid function at x.
        :param x: Vector.
        :return: Value returned.
        """
        return  1.0 - 1.0/(1.0 + math.exp(sum(w*x)))

    def ComputeCoeffs(self,data, features, label, alpha, iters, threshold, reg, numFrag):
        """
        Perform a logistic regression via gradient ascent.
        """

        theta = theta = np.array(np.zeros(3), dtype = float)   #initial
        i = 0
        converged = False
        threshold = 0.001
        reg = 0
        while ( (i<iters) and not converged):

            # grEin = gradient of in-sample Error
            grEin = [self.GradientAscent(data[f],features,label,theta,alpha) for f in range(numFrag)]
            grad  = mergeReduce(self.agg_sga,grEin)
            theta, converged = self.calcTheta(grad,alpha,i,reg,threshold)
            i+=1
            converged = compss_wait_on(converged)
        theta = compss_wait_on(theta)

        return theta


    @task(returns=list, isModifier = False)
    def GradientAscent(self,data,X,Y,theta,alfa):
        # Estimate logistic regression coefficients using stochastic gradient descent
        if isinstance(data.iloc[0][X], list):
            dim = len(data.iloc[0][X])
        else:
            dim = 1

        if (dim+1) != len(theta):
            theta = np.array(np.zeros(dim+1), dtype = float)

        N = len(data)
        # get the sum of error in the whole se
        gradient = 0

        Xs = np.c_[np.ones(N), np.array(data[X].tolist() ) ] # adding ones

        for n in range(N):
            xn = np.array(Xs[n, :])
            yn = data[Y].values[n]
            grad_p = self.sigmoid(xn, theta)
            gradient += xn*(yn - grad_p)

        return [gradient, N, dim, theta]

    @task(returns=list, isModifier = False)
    def agg_sga(self,info1,info2):
        #print "{} + {} = {}".format(info1[0],info2[0],info1[0]+info2[0])
        return [info1[0]+info2[0], info2[1]+info2[1], info1[2], info1[3]]


    @task(returns=(float,bool), isModifier = False)
    def calcTheta(self,info,coef_lr,it, regularization,threshold):
        gradient  = info[0]
        N     = info[1]
        dim   = info[2]
        theta = info[3]

        # update coefficients
        alpha = coef_lr/(1+it)
        theta += alpha*(gradient - regularization*theta)

        converged = False
        if alpha*sum(gradient*gradient) < threshold:
    	       converged = True
        return theta,converged


    def transform(self,data,model,settings,numFrag):
        col_features = settings['features']
        predCol    = settings.get('predCol','prediction')


        data = [ self.predict(data[f],col_features,predCol,model)
                                                        for f in range(numFrag)]

        return data

    @task(returns=list, isModifier = False)
    def predict(self,data,X,predCol,theta):
        N = len(data)

        Xs = np.c_[np.ones(N), np.array(data[X].tolist() ) ]
        data[predCol] = [ round(self.sigmoid(x,theta)) for x in Xs] #1 if x >= 0.5 else 0
        return data
