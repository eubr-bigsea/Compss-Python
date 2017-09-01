#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce


class Kmeans(object):
    """

    
    """



    def fit(self,data,settings,numFrag):
        """
            fit():

            - :param data:        A list with numFrag pandas's dataframe used to create the model.
            - :param settings:    A dictionary that contains:
             	- k:  			  Number of wanted clusters.
             	- features: 	  Column name of the features in the dataset;
             	- maxIterations:  Maximum number of iterations;
                - epsilon:        Threshold to stop the iterations;
                - initMode:       "random" or "k-means||"
            - :param numFrag:     A number of fragments;
            - :return:            The model created (which is a pandas dataframe).
        """
        from pycompss.api.api import compss_wait_on

        features_col = settings['features']

        k             = int(settings['k'])
        maxIterations = int(settings['maxIterations'])
        epsilon       = float(settings['epsilon'])
        initMode      = settings['initMode']

        if initMode == "random":
            from random import randint
            partitions_C = [randint(0, numFrag-3) for i in range(k)]
            centroids = [ self.initMode_random(data[f], features_col, f, partitions_C) for f in range(numFrag)]
            centroids = mergeReduce(self.mergeCentroids,centroids)
            centroids = compss_wait_on(centroids)
        elif initMode == "k-means||":
            centroids = self.init_parallel(data, features_col, k, numFrag)

        size = centroids[1]
        centroids = centroids[0]

        old_centroids = []
        it = 0

        while not self.has_converged(centroids, old_centroids, epsilon, it, maxIterations):
            old_centroids = list(centroids)
            idx = [ self.find_closest_centroids(data[f], features_col, centroids) for f in range(numFrag)]

            idx = mergeReduce(self.reduceCentersTask, idx)
            idx = compss_wait_on(idx)
            centroids = self.compute_centroids(idx, numFrag)

            it += 1
            #print "Iter:{} - Centroids:{}".format(it,centroids)

        model = pd.DataFrame([[c] for c in centroids],columns=["Clusters"])

        return model

    def transform(self, data, model, settings, numFrag):
        """
            transform():

            - :param data:       A list with numFrag pandas's dataframe that will be predicted.
            - :param model:		 The Kmeans model created;
            - :param settings:    A dictionary that contains:
             	- features: 	  Column name of the features in the test data;
             	- predCol:    	  Alias to the new column with the labels predicted;
            - :param numFrag:     A number of fragments;
            - :return:            The prediction (in the same input format).
        """

        features_col = settings['features']
        predCol     = settings.get('predCol','Prediction')

        data = [self.assigment_cluster(data[f], features_col, model, predCol) for f in range(numFrag) ]
        return data

    @task(returns=dict,isModifier = False)
    def find_closest_centroids(self,data, columns, mu):

        XP = np.array(data[columns].values)
        new_centroids = dict()
        k = len(mu)

        for i in range(k):
            new_centroids[i] = [0,[]]

        for x in XP:
            distances = np.array([ np.linalg.norm(x - np.array(mu[j]) ) for j in range(k) ])
            bestC = distances.argmin(axis=0)
            new_centroids[bestC][0]+=1
            new_centroids[bestC][1].append(x)


        for i in range(k):
            new_centroids[i][1] = np.sum(new_centroids[i][1], axis=0)

        #print new_centroids
        return new_centroids

    @task(returns=dict, priority=True,isModifier = False)
    def reduceCentersTask(self,a, b):
        for key in b:
            if key not in a:
                a[key] = b[key]
            else:
                a[key] = (a[key][0] + b[key][0], a[key][1] + b[key][1])
        return a

    def compute_centroids(self, centroids, numFrag):
        """ Next we need a function to compute the centroid of a cluster.
            The centroid is simply the mean of all of the examples currently
            assigned to the cluster."""


        centroids = [ centroids[c][1] / centroids[c][0] for c in centroids]
        return centroids

    @task(returns=list,isModifier = False)
    def assigment_cluster(self,data, columns, model, predCol):
        mu = model['Clusters'].tolist()
        XP = np.array(data[columns].values)
        k = len(mu)
        values = []
        for x in XP:
            distances = np.array([ np.linalg.norm(x - mu[j] ) for j in range(k) ])
            bestC = distances.argmin(axis=0)
            values.append(bestC)

        data[predCol] = pd.Series(values).values
        return data

    def has_converged(self,mu, oldmu, epsilon, iter, maxIterations):
        if oldmu != []:
            if iter < maxIterations:
                aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
                distancia = sum(aux)
                if distancia < (epsilon**2):
                    return True
                else:
                    return False
            else:
                return True




    @task(returns=list,isModifier = False)
    def cost(self, data, columns, C):
        XP = np.array(data[columns].values)
        dist = 0
        for x in XP:
            dist+= self.distance(x, C)**2

        return dist

    def distance(self,x, C):
        return min( np.array([ np.linalg.norm(x - np.array(c) ) for c in C ]) )

    @task(returns=list,isModifier = False)
    def mergeCostPhi(self, cost1, cost2):
        return cost1+cost2

    @task(returns=list,isModifier = False)
    def initC(self, data,columns):
        import random
        return [ random.sample(data[columns].tolist(), 1),0]


    def init_parallel(self, data, columns, k,  numFrag):
        """
        Initialize a set of cluster centers using the k-means|| algorithm by Bahmani et al.
        (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
        to find dissimilar cluster centers by starting with a random center and then doing
        passes where more centers are chosen with probability proportional to their squared distance
        to the current cluster set. It results in a provable approximation to an optimal clustering.

        The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.

        l = oversampling factor =  0.2k or 1.5k
        """
        from pycompss.api.api import compss_wait_on
        l = 0.2*k

        """
        Step1: C ← sample a point uniformly at random from X
        """
        import random

        f = random.randint(0, numFrag-1)
        C = self.initC(data[f], columns )

        """
        Step2: Compute ϕX(C) as the sum of all distances from xi to the closest point in C
        """
        cost_p = [ self.cost(data[f],columns, C) for f in range(numFrag)] #compute d2 for each x_i
        phi = mergeReduce(self.mergeCostPhi, cost_p)
        phi = compss_wait_on(phi)
        LogPhi = int(round(np.log(phi)))
        #print "cost", phi
        #print "phi", LogPhi
        #print "L",l
        #print "iter:{} | C:{}".format(-1,C[0])
        for i in range(LogPhi):
            """
            Step3: For each point in X, denoted as xi, compute a probability from
            px=l*d2(x,C)/ϕX(C). Here you have l a factor given as parameter,
            d2(x,C) is the distance to the closest center, and ϕX(C) is explained
            at step 2.
            """
            c = [self.probabilities(data[f], columns, C, l, phi,f) for f in range(numFrag)]
            #print c
            C = mergeReduce(self.mergeCentroids,c)
            #print "iter:{} | C:{}".format(i,C[0])

        """
        In the end we have C with all centroid candidates

        Step 4: pick k centers. A simple algorithm for that is to create a
        vector w of size equals to the number of elements in C, and initialize
        all its values with 0. Now iterate in X (elements not selected in as
        centroids), and for each xi∈X, find the index j of the closest centroid
        (element from C) and increment w[j]] with 1.

        Step 5: when exactly k centers have been chosen, finalize the
        initialization phase and proceed with the standard k-means algorithm

        """
        w = [ self.bestMuKey(data[f], columns, C) for f in range(numFrag) ]
        ws = mergeReduce(self.MergeBestMuKey, w)
        ws = compss_wait_on(ws)

        Centroids = np.argsort(ws)
        best_id =  Centroids[:k]
        #print best_id
        C = compss_wait_on(C)
        Centroids = [ C[0][index] for index in best_id]

        #print "Centroids",Centroids
        return Centroids


    @task(returns=list,isModifier = False)
    def probabilities(self, data, columns, C, l, phi,f):
        newC = []

        for x in data[columns].values:
            if x not in C[0]:
                px = (l*self.distance(x, C[0])**2)/phi
                if px >= np.random.random(1):
                    newC.append(x)


        #    print "newC-input",newC
        #print "C",C[0]
        if f == 0 :
            if len(newC) == 0:
                newC = C[0]
            else:
                newC = newC + C[0]

        #print "newC-output",newC
        return [newC, len(data)]

    @task(returns=list,isModifier = False)
    def MergeBestMuKey(self,w1,w2):
        for i in range(len(w2)):
            w1[i]+=w2[i]


        return w1

    @task(returns=list,isModifier = False)
    def bestMuKey(self,data,columns, C):
        C = C[0]
        w = [0 for i in xrange(len(C))]
        for x in np.array(data[columns]):
            distances = np.array([ np.linalg.norm(x - np.array(c) ) for c in C ])
            bestC = distances.argmin(axis=0)
            w[bestC]+=1

        return w




    @task(returns=list,isModifier = False)
    def initMode_random(self,data, columns,  f, partitions_C):
        from random import sample
        num = sum([1 for c in partitions_C if c == f ])
        sampled = sample(data[columns].tolist(), num)
        size = len(data)

        return [sampled,[size]]


    @task(returns=list,isModifier = False)
    def mergeCentroids(self,a, b):
        return [ a[0] + b[0] , a[1] + b[1]]
