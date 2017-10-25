#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce


import math
import numpy as np

class linearRegression(object):

    """

    Linear regression is a linear model, e.g. a model that assumes a linear
    relationship between the input variables and the single output variable.
    More specifically, that y can be calculated from a linear combination of the
    input variables (x).

    When there is a single input variable (x), the method is referred to as
    simple linear regression. When there are multiple input variables,
    literature from statistics often refers to the method as multiple
    linear regression.

    b1 = (sum(x*y) + n*m_x*m_y) / (sum(x²) -n*(m_x²))
    b0 = m_y - b1*m_x

    Methods:
        - fit()
        - transform()

    """

    def fit(self,data,settings,numFrag):
        """
            fit():

            - :param data:      A list with numFrag pandas's dataframe
                                used to create the model.
            - :param settings:  A dictionary that contains:
             	- features: 	Field of the features in the dataset;
                - label: 	    Field of the label in the dataset;
                - mode:
                    * 'simple': Best option if is a 2D regression;
                    * 'SDG':    Uses a Stochastic gradient descent to perform
                                the regression. Can be used to data of all
                                dimensions.
             	- max_iter:     Maximum number of iterations, only using 'SDG'
                                (integer, default: 100);
                - alpha:        Learning rate parameter, only using 'SDG'
                                (float, default 0.01)
            - :param numFrag:   A number of fragments;
            - :return:          Returns a model (which is a pandas dataframe).

            Note: Best results with a normalizated data.
        """


        features = settings['features']
        label    = settings['label']
        mode = settings.get('mode', 'SDG')

        if mode not in ['simple','SDG']:
           raise Exception("You must inform a valid `mode`.")

        if mode == "SDG":
            alpha = settings.get('alpha', 0.1)
            iters = settings.get('max_iter', 100)

            parameters = gradientDescent(  data, features, label,
                                                alpha, iters, numFrag)
        elif mode == 'simple':
            """
                Simple Linear Regression: This mode is useful only if
                you have a small dataset.
            """

            xs = [calcsXs(data[f], features) for f in range(numFrag)]
            ys = [calcsXs(data[f], label)    for f in range(numFrag)]
            xys= [calcsXYs(data[f],features,label) for f in range(numFrag)]

            rx  = mergeReduce(mergeCalcs,xs)
            ry  = mergeReduce(mergeCalcs,ys)
            rxy = mergeReduce(mergeCalcs,xys)

            parameters = computeLine2D(rx,ry,rxy)


        model = dict()
        model['algorithm'] = 'linearRegression'
        model['model'] = parameters
        return model




    def transform(self, data, model, settings, numFrag):
        """
            transform():

            - :param data:      A list with numFrag pandas's dataframe
                                that will be predicted.
            - :param model:		The Linear Regression's model created;
            - :param settings:  A dictionary that contains:
             	- features: 	Field of the features in the test data;
             	- predCol:    	Alias to the new predicted labels;
            - :param numFrag:   A number of fragments;
            - :return:          The prediction (in the same input format).

        """

        if 'features' not in settings:
           raise Exception("You must inform the `features` field.")

        features = settings['features']
        predCol  = settings.get('predCol','PREDICTED_VALUE')

        if model.get('algorithm','null') != 'linearRegression':
            raise Exception("You must inform a valid Linear Regression model.")

        model = model['model']
        result = [[] for f in range(numFrag)]
        for f in range(numFrag):
            result[f] = predict(data[f], features, predCol, model)

        return result

# --------------
# Simple Linear Regression

@task(returns=list)
def calcsXs(X,col):
    sumX = X[col].sum()
    squareX= (X[col]**2).sum()
    #print "sum:{} - Square:{}".format(sumX,squareX)
    return [sumX, len(X[col]),squareX]


@task(returns=list)
def calcsXYs(XY,col1,col2):
    r = sum([x*y for x,y in zip(XY[col1],XY[col2])])
    return [0,0,r]


@task(returns=list)
def mergeCalcs(p1,p2):
    sumT = p1[0] + p2[0]
    T = p1[1] + p2[1]
    squareT = p1[2] + p2[2]
    return [sumT,T,squareT]


@task(returns=list)
def computeLine2D(rx,ry,rxy):
    n = rx[1]
    m_x = (float(rx[0])/n)
    m_y = (float(ry[0])/n)
    b1  =  float(rxy[2] - n*m_x*m_y) /(rx[2] - rx[1]* (m_x**2))
    b0  = m_y - b1*m_y

    return [b0, b1]




# --------------
# SGD mode:

def gradientDescent(data, features, label, alpha, iters, numFrag):
    theta = np.array([0,0,0])

    #cost = np.zeros(iters)

    for i in range(iters):
        stage1 = [firststage(data[f],features,label,theta)
                    for f in range(numFrag)]
        grad  = mergeReduce(agg_SGD,stage1)
        theta = calcTheta(grad, alpha)

        #cost[i] = [computeCost(data[f],features,label, theta) for f in range(numFrag)]
        #theta = compss_wait_on(theta)

    return theta#, cost

@task(returns=list)
def firststage(data,X,Y,theta):
    N = len(data)

    if N >0:
        if isinstance(data.iloc[0][X], list):
            dim = len(data.iloc[0][X])
        else:
            dim = 1

        if (dim+1) != len(theta):
            theta = np.array([0 for i in range(dim+1)])

        Xs = np.c_[np.ones(N), np.array(data[X].tolist() ) ]
        partial_error = np.dot(Xs, theta.T) - data[Y].values

        for j in range(dim+1):
            grad = np.multiply(partial_error, Xs[:,j])

        return [np.sum(grad), N, dim, theta]

    return [0, 0, -1, 0]

@task(returns=list)
def agg_SGD(error1,error2):
    dim1 = error1[2]
    dim2 = error2[2]

    if dim1 > 0:
        sum_grad = error1[0]+error2[0]
        N = error2[1]+error2[1]
        dim = dim1
        theta = error1[3]
    elif dim2 > 0:
        sum_grad = error1[0]+error2[0]
        N = error2[1]+error2[1]
        dim = dim2
        theta = error2[3]
    else:
        sum_grad = 0
        N = 0
        dim = -1
        theta = 0

    return [sum_grad, N, dim, theta]


@task(returns=list)
def calcTheta(info, alpha):
    grad  = info[0]
    N     = info[1]
    dim   = info[2]
    theta = info[3]

    temp = np.zeros(theta.shape)
    for j in range(dim+1):
        temp[j] = theta[j] - ((float(alpha) / N) * grad)

    return temp


@task(returns=list)
def predict(data,X,Y,model):

    tmp = []
    if len(data)>0:
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
        else:
            tmp = [model[0] + model[1]*row for row in data[X].values]

    data[Y] = tmp
    return data
