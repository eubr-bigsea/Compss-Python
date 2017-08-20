#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.functions.data    import chunks
from pycompss.api.api import compss_wait_on
import math
import numpy as np

class linearRegression(object):

    """

    Linear regression is a linear model, e.g. a model that assumes a linear
    relationship between the input variables (x) and the single output variable (y).
    More specifically, that y can be calculated from a linear combination of the
    input variables (x).

    When there is a single input variable (x), the method is referred to as simple
    linear regression. When there are multiple input variables, literature from
    statistics often refers to the method as multiple linear regression.

    b1 = (sum(x*y) + n*m_x*m_y) / (sum(x²) -n*(m_x²))
    b0 = m_y - b1*m_x

    """

    @task(returns=list, isModifier = False)
    def calcsXs(self,X,col):
        sumX = X[col].sum()
        squareX= (X[col]**2).sum()
        #print "sum:{} - Square:{}".format(sumX,squareX)
        return [sumX, len(X[col]),squareX]


    @task(returns=list, isModifier = False)
    def calcsXYs(self,XY,col1,col2):
        r = sum([x*y for x,y in zip(XY[col1],XY[col2])])
        return [0,0,r]


    @task(returns=list, isModifier = False)
    def mergeCalcs(self,p1,p2):
        sumT = p1[0] + p2[0]
        T = p1[1] + p2[1]
        squareT = p1[2] + p2[2]
        return [sumT,T,squareT]


    @task(returns=list, isModifier = False)
    def computeLine2D(self,rx,ry,rxy):
        n = rx[1]
        m_x = (float(rx[0])/n)
        m_y = (float(ry[0])/n)
        b1  =  float(rxy[2] - n*m_x*m_y) /(rx[2] - rx[1]* (m_x**2))
        b0  = m_y - b1*m_y
        #SST = ry[2] - n*(m_y**2)
        return [b0, b1]



    def fit(self,data,settings,numFrag):
        features = settings['features']
        label    = settings['label']


        if settings['option'] != 'SDG' and settings['dim'] == "2D":
            """ Simple Linear Regression """
            xs  = [ self.calcsXs(data[f], features) for f in range(numFrag) ]
            ys  = [ self.calcsXs(data[f],label)    for f in range(numFrag)  ]
            xys = [ self.calcsXYs(data[f],features,label)
                                                    for f in range(numFrag) ]
            rx  =  mergeReduce(self.mergeCalcs,xs)
            ry  =  mergeReduce(self.mergeCalcs,ys)
            rxy =  mergeReduce(self.mergeCalcs,xys)

            parameters = self.computeLine2D(rx,ry,rxy)
        else:
            alpha = settings['alpha']
            iters = settings['iters']

            parameters = self.gradientDescent(data, features, label, alpha, iters, numFrag)

        return parameters




    def transform(self,data,settings,numFrag):
        features = settings['features']
        label    = settings['new_label']
        model    = settings['model']

        data = [self.predict(data[f],features,label,model) for f in range(numFrag)]

        return data

    @task(returns=list, isModifier = False)
    def predict(self,data,X,Y,model):

        tmp = []
        if isinstance(data.iloc[0][X], list):
            dim = len(data.iloc[0][X])
        else:
            dim = 1
        if dim >1:
            for row in data[X].values:
                y = row[0]
                for j in xrange(1,len(row)):
                    y += row[j]*model[j]
                tmp.append(y)
            data[Y] = tmp
        else:
            data[Y] = [model[0] + model[1]*row for row in data[X].values]
        return data



    def gradientDescent(self,data, features, label, alpha, iters, numFrag):
        theta = np.array([0,0,0])   #initial

        #cost = np.zeros(iters)

        for i in range(iters):
            stage1 = [self.firststage(data[f],features,label,theta) for f in range(numFrag)]
            grad  = mergeReduce(self.agg_SGD,stage1)
            theta = self.calcTheta(grad, alpha)
            #cost[i] = [self.computeCost(data[f],features,label, theta) for f in range(numFrag)]
            #theta = compss_wait_on(theta)

        return theta#, cost

    @task(returns=list, isModifier = False)
    def firststage(self,data,X,Y,theta):
        if isinstance(data.iloc[0][X], list):
            dim = len(data.iloc[0][X])
        else:
            dim = 1
        #print dim
        #print len(theta)
        if (dim+1) != len(theta):
            theta = np.array([0 for i in range(dim+1)])

        N = len(data)

        Xs = np.c_[np.ones(N), np.array(data[X].tolist() ) ]
        partial_error = np.dot(Xs, theta.T) - data[Y].values

        for j in range(dim+1):
            grad = np.multiply(partial_error, Xs[:,j])

        return [np.sum(grad), N, dim, theta]

    @task(returns=list, isModifier = False)
    def agg_SGD(self,error1,error2):
        return [error1[0]+error2[0], error2[1]+error2[1], error1[2], error1[3]]


    @task(returns=list, isModifier = False)
    def calcTheta(self,info, alpha):
        grad  = info[0]
        N     = info[1]
        dim   = info[2]
        theta = info[3]

        temp = np.zeros(theta.shape)
        for j in range(dim+1):
            temp[j] = theta[j] - ((float(alpha) / N) * grad)

        return temp

    # @task(returns=list, isModifier = False) # TO DO
    # def computeCost(X, y, theta):
    #     inner = np.power(((X * theta.T) - y), 2)
    #     return np.sum(inner) / (2 * len(X))
