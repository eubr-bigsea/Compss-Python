#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce

from pycompss.api.api import compss_wait_on
import math
import numpy as np

class logisticRegression(object):

    """
    Logistic regression is named for the function used at the core
    of the method, the logistic function. It is the go-to method for
    binary classification problems (problems with two class values).

    The logistic function, also called the sigmoid function was
    developed by statisticians to describe properties of population
    growth in ecology, rising quickly and maxing out at the carrying
    capacity of the environment. It’s an S-shaped curve that can take
    any real-valued number and map it into a value between 0 and 1,
    but never exactly at those limits.

    This implementation uses a Stochastic Gradient Ascent (a variant of
    the Stochastic gradient descent). It is called stochastic because
    the derivative based on a randomly chosen single example is a random
    approximation to the true derivative based on all the training data.
    """


    def fit(self,data,settings,numFrag):

        """
        fit():

        :param data:        A list with numFrag pandas's dataframe used to training the model.
        :param settings:    A dictionary that contains:
            - iters:            Maximum number of iterations (integer);
            - threshold:        Tolerance for stopping criterion (float);
            - regularization:   Regularization parameter (float);
            - alpha:            The Learning rate, it means, how large of steps to take
                                on our cost curve (float);
         	- features: 		Column name of the features in the training data;
         	- label:          	Column name of the labels   in the training data;
        :param numFrag:     A number of fragments;
        :return:            The model created (which is a pandas dataframe).
        """

        features = settings['features']
        label    = settings['label']
        alpha = settings['alpha']
        iters = settings['iters']
        threshold = settings['threshold']
        reg = settings['regularization']
        parameters = self.ComputeCoeffs(data, features, label, alpha,
                                                iters, threshold, reg, numFrag)

        return parameters


    def transform(self,data,model,settings,numFrag):
        """
        transform():

        :param data:        A list with numFrag pandas's dataframe that will be predicted.
        :param model:		 The Logistic Regression model created;
        :param settings:    A dictionary that contains:
 	      - features: 		 Column name of the features in the test data;
 	      - predCol:    	 Alias to the new column with the labels predicted;
        :param numFrag:     A number of fragments;
        :return:            The prediction (in the same input format).
        """
        col_features = settings['features']
        predCol    = settings.get('predCol','prediction')

        data = [ self.predict(data[f],col_features,predCol,model) for f in range(numFrag)]

        return data


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




    @task(returns=list, isModifier = False)
    def predict(self,data,X,predCol,theta):
        N = len(data)

        Xs = np.c_[np.ones(N), np.array(data[X].tolist() ) ]
        data[predCol] = [ round(self.sigmoid(x,theta)) for x in Xs] #1 if x >= 0.5 else 0
        return data
