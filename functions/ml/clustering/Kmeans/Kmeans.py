#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce


class Kmeans(object):
    def fit(self, settings):
        """
            Kmeans.fit():
                Setting up the model.


            :param k: A number of centroids
            :param maxIterations: max iterations
            :param epsilon: error threshold
            :return A model
        """
        k = settings['k']
        maxIterations = settings['maxIterations']
        epsilon = settings['epsilon']
        initMode = settings['initMode']

        return [k,maxIterations,epsilon,initMode]


    def transform(self,data,model,numFrag):
        """
            kmeans: starting with a set of randomly chosen initial centers,
            one repeatedly assigns each imput point to its nearest center, and
            then recomputes the centers given the point assigment. This local
            search called Lloyd's iteration, continues until the solution does
            not change between two consecutive rounds or iteration > maxIterations.

            :param model: A model with the configurations
            :param data:  A np.array (splitted)
            :return: list os centroids
        """
        from pycompss.api.api import compss_wait_on

        k             = int(model[0])
        maxIterations = int(model[1])
        epsilon       = float(model[2])
        initMode      = model[3]

        mu = self.init(data, k, initMode)
        oldmu = []
        n = 0
        size = int(len(data) / numFrag)

        while not self.has_converged(mu, oldmu, epsilon, n, maxIterations):
            oldmu = list(mu)
            clusters = [self.cluster_points_partial(data[f], mu, f * size) for f in range(numFrag)]
            partialResult = [self.partial_sum(data[f], clusters[f], f * size) for f in range(numFrag)]

            mu = mergeReduce(self.reduceCentersTask, partialResult)
            mu = compss_wait_on(mu)
            mu = [mu[c][1] / mu[c][0] for c in mu]
            n += 1
        return mu

    @task(returns=dict,isModifier = False)
    def cluster_points_partial(self,XP, mu, ind):
        dic = {}
        XP = np.array(XP)
        for x in enumerate(XP):
            bestmukey = min([(i[0], np.linalg.norm(x[1] - mu[i[0]]))
                             for i in enumerate(mu)], key=lambda t: t[1])[0]
            if bestmukey not in dic:
                dic[bestmukey] = [x[0] + ind]
            else:
                dic[bestmukey].append(x[0] + ind)
        return dic


    @task(returns=dict,isModifier = False)
    def partial_sum(self,XP, clusters, ind):
        XP = np.array(XP)
        p = [(i, [(XP[j - ind]) for j in clusters[i]]) for i in clusters]
        dic = {}
        for i, l in p:
            dic[i] = (len(l), np.sum(l, axis=0))
        return dic


    @task(returns=dict, priority=True,isModifier = False)
    def reduceCentersTask(self,a, b):
        for key in b:
            if key not in a:
                a[key] = b[key]
            else:
                a[key] = (a[key][0] + b[key][0], a[key][1] + b[key][1])
        return a


    def has_converged(self,mu, oldmu, epsilon, iter, maxIterations):
        if oldmu != []:
            if iter < maxIterations:
                aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
                distancia = sum(aux)
                if distancia < epsilon * epsilon:
                    return True
                else:
                    return False
            else:
                return True


    def distance(self,p, X):
        return min([np.linalg.norm(np.array(p)-x) for x in X])


    def cost(self,Y, C):
        return sum([self.distance(x, C)**2 for x in Y])


    def bestMuKey(self,X, C):
        w = [0 for i in xrange(len(C))]
        for x in X:
            bestmukey = min([(i[0], np.linalg.norm(x-np.array(C[i[0]])))
                            for i in enumerate(C)], key=lambda t: t[1])[0]
            w[bestmukey] += 1
        return w



    def probabilities(self,X, C, l, phi, n):
        np.random.seed(5)
        p = [(l*self.distance(x, C)**2)/phi for x in X]
        p /= sum(p)
        idx = np.random.choice(n, l, p=p)
        newC = [X[i][0] for i in idx]
        return newC



    def init_parallel(self,X, k, l, initSteps=2):
        import random
        random.seed(5)
        numFrag = len(X)
        ind = random.randint(0, numFrag-1)
        XP  = X[ind]
        C = random.sample(XP, 1)
        phi = sum([self.cost(x, C) for x in X])

        for i in range(initSteps):
            '''calculate p'''
            c = [self.probabilities(x, C, l, phi, len(x)) for x in X]
            C.extend([item for sublist in c for item in sublist])
            '''cost distributed'''
            phi = sum([self.cost(x, C) for x in X])

        '''pick k centers'''
        w = [self.bestMuKey(x, C) for x in X]
        bestC = [sum(x) for x in zip(*w)]
        bestC = np.argsort(bestC)
        bestC = bestC[::-1]
        bestC = bestC[:k]
        return [C[b] for b in bestC]


    def init_random(self,dim, k):
        np.random.seed(2)
        m = np.array([np.random.random(dim) for _ in range(k)])
        return m


    def init(self,X, k, mode):
        if mode == "kmeans++":
            return self.init_parallel(X, k, k)
        else:
            dim = len(X[0][0])
            return self.init_random(dim, k)
