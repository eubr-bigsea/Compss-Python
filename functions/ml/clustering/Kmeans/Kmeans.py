#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

import numpy as np
import pandas as pd

from pycompss.api.parameter    import *
from pycompss.api.task         import task
from pycompss.functions.reduce import mergeReduce


class Kmeans(object):

    def fit(self, data, settings, numFrag):
        """
            fit():

            - :param data:        A list with numFrag pandas's dataframe
                                  used to create the model.
            - :param settings:    A dictionary that contains:
             	- k:  			  Number of wanted clusters.
             	- features: 	  Field of the features in the dataset;
             	- maxIterations:  Maximum number of iterations;
                - epsilon:        Threshold to stop the iterations;
                - initMode:       "random" or "k-means||"
            - :param numFrag:     A number of fragments;
            - :return:            Returns a model (which is a pandas dataframe).
        """

        k             = int(settings.get('k', 2))
        maxIterations = int(settings.get('maxIterations', 100))
        epsilon       = float(settings.get('epsilon', 0.001))
        initMode      = settings.get('initMode','k-means||')

        if initMode not in ['random','k-means||']:
            raise Exception("This implementation only suports 'random' and "
                            "k-means|| as a initialization mode. ")

        if 'features' not in settings:
           raise Exception("You must inform the `features` field.")

        features_col  = settings['features']


        from pycompss.api.api import compss_wait_on
        #counting rows in each fragment
        size = [getSize(data[f]) for f in range(numFrag)]
        size = mergeReduce(mergeCentroids, size)
        size = compss_wait_on(size)
        size, n_list = size

        if initMode == "random":
            centroids = init_random( data, features_col,
                                          k, size, n_list, numFrag)


        elif initMode == "k-means||":
            centroids = init_parallel( data, features_col,
                                            k, size, n_list, numFrag)

        old_centroids = []
        it = 0

        print "[INFO] - Inital clusters: ", centroids


        while not has_converged(centroids, old_centroids, epsilon,
                                it, maxIterations):
            old_centroids = list(centroids)
            idx = [
                   find_closest_centroids(data[f], features_col, centroids)
                   for f in range(numFrag)
                    ]

            idx = mergeReduce(reduceCentersTask, idx)
            idx = compss_wait_on(idx)
            centroids = compute_centroids(idx, numFrag)
            it += 1

        model = {}
        model['algorithm'] = 'K-Means'
        model['model'] = pd.DataFrame(  [[c] for c in centroids],
                                        columns=["Clusters"])


        return model

    def transform(self, data, model, settings, numFrag):
        """
            transform():

            - :param data:      A list with numFrag pandas's dataframe
                                that will be predicted.
            - :param model:		The Kmeans model created;
            - :param settings:  A dictionary that contains:
             	- features: 	Field of the features in the test data;
             	- predCol:    	Alias to the new predicted labels;
            - :param numFrag:   A number of fragments;
            - :return:          The prediction (in the same input format).
        """

        if 'features' not in settings:
           raise Exception("You must inform the `features` field.")

        if model.get('algorithm','null') != 'K-Means':
            raise Exception("You must inform a valid model.")

        model = model['model']
        features_col = settings['features']
        predCol      = settings.get('predCol','Prediction')

        result = [[] for f in range(numFrag)]
        for f in range(numFrag):
            result[f] = assigment_cluster(data[f], features_col, model, predCol)

        return result

@task(returns=list)
def getSize(data):
    return [len(data),[len(data)]]

@task(returns=list)
def mergeCentroids(a, b):
    return [ a[0] + b[0] , a[1] + b[1]]

@task(returns=dict)
def find_closest_centroids(data, columns, mu):

    XP = np.array(data[columns].values)
    new_centroids = dict()
    k = len(mu)

    for i in range(k):
        new_centroids[i] = [0,[]]

    for x in XP:
        distances = \
        np.array([ np.linalg.norm(x - np.array(mu[j]) ) for j in range(k) ])
        bestC = distances.argmin(axis=0)
        new_centroids[bestC][0]+=1
        new_centroids[bestC][1].append(x)


    for i in range(k):
        new_centroids[i][1] = np.sum(new_centroids[i][1], axis=0)

    #print new_centroids
    return new_centroids

@task(returns=dict, priority=True)
def reduceCentersTask(a, b):
    for key in b:
        if key not in a:
            a[key] = b[key]
        else:
            a[key] = (a[key][0] + b[key][0], a[key][1] + b[key][1])
    return a

def compute_centroids( centroids, numFrag):
    """ Next we need a function to compute the centroid of a cluster.
        The centroid is simply the mean of all of the examples currently
        assigned to the cluster."""

    centroids = [ centroids[c][1] / centroids[c][0] for c in centroids]
    return centroids

@task(returns=list)
def assigment_cluster(data, columns, model, predCol):
    mu = model['Clusters'].tolist()
    XP = np.array(data[columns].values)
    k = len(mu)
    values = []
    for x in XP:
        distances = np.array([np.linalg.norm(x - mu[j] ) for j in range(k)])
        bestC = distances.argmin(axis=0)
        values.append(bestC)

    data[predCol] = pd.Series(values).values
    return data

def has_converged(mu, oldmu, epsilon, iter, maxIterations):
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


@task(returns=list)
def cost( data, columns, C):
    XP = np.array(data[columns].values)
    dist = 0
    for x in XP:
        dist+= distance(x, C)**2

    return dist

def distance(x, C):
    return min( np.array([ np.linalg.norm(x - np.array(c) ) for c in C ]) )

@task(returns=list)
def mergeCostPhi( cost1, cost2):
    return cost1+cost2


# ------ INIT MODE: random
def init_random( data, features_col, k, size, n_list, numFrag):
    from pycompss.api.api import compss_wait_on

    #define where get the inital core.
    ids = sorted(np.random.choice(size, k, replace=False))
    list_ids = [[] for i in range(numFrag)]

    frag = 0
    maxIdFrag = n_list[frag]
    oldmax = 0
    for i in ids:
        while i >= maxIdFrag:
            frag+=1
            oldmax = maxIdFrag
            maxIdFrag+= n_list[frag]

        list_ids[frag].append(i-oldmax)

    fs_valids = [f for f in range(numFrag) if len(list_ids[f]) > 0]

    centroids = [ initMode_random(data[f], features_col, list_ids[f])
                    for f in fs_valids ]
    centroids = mergeReduce(mergeCentroids,centroids)

    centroids = compss_wait_on(centroids)

    return centroids

@task(returns=list)
def initMode_random(data, columns, idxs):
    sampled = data.loc[idxs, columns].tolist()
    return [sampled, [0]]

# ------ INIT MODE: k-means||
def init_parallel( data, columns, k, size, n_list, numFrag):
    """
    Initialize a set of cluster centers using the k-means|| algorithm
    by Bahmani et al. (Bahmani et al., Scalable K-Means++, VLDB 2012).
    This is a variant of k-means++ that tries to find dissimilar cluster
    centers by starting with a random center and then doing passes where
    more centers are chosen with probability proportional to their squared
    distance to the current cluster set. It results in a provable
    approximation to an optimal clustering.

    The original paper can be found at:
    http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.

    l = oversampling factor =  0.2k or 1.5k
    """
    from pycompss.api.api import compss_wait_on
    l = 0.2*k

    """
    Step1: C ← sample a point uniformly at random from X
    """
    fs_valids = [f for f in range(numFrag) if n_list[f] > 0]
    f = np.random.choice(fs_valids,1,replace=False)[0]
    C = initC(data[f], columns )

    """
    Step2:  Compute ϕX(C) as the sum of all distances from xi to the
            closest point in C
    """
    cost_p = [ cost(data[f],columns, C) for f in range(numFrag)]
    phi = mergeReduce(mergeCostPhi, cost_p)
    phi = compss_wait_on(phi)
    LogPhi = int(round(np.log(phi)))

    for i in range(LogPhi):
        """
        Step3:  For each point in X, denoted as xi, compute a probability
                from px=l*d2(x,C)/ϕX(C). Here you have l a factor given
                as parameter, d2(x,C) is the distance to the closest center,
                and ϕX(C) is explained at step 2.
        """
        c = [ probabilities(data[f], columns, C, l, phi,f)
                for f in range(numFrag)]

        C = mergeReduce(mergeCentroids,c)

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
    w = [ bestMuKey(data[f], columns, C) for f in range(numFrag) ]
    ws = mergeReduce(MergeBestMuKey, w)
    ws = compss_wait_on(ws)

    Centroids = np.argsort(ws)
    best_id =  Centroids[:k]

    C = compss_wait_on(C)
    Centroids = [ C[0][index] for index in best_id]

    return Centroids

@task(returns=list)
def initC( data, columns):
    sample = data.sample(n=1)
    sample = sample[columns].tolist()
    return [sample, 0]

@task(returns=list)
def probabilities( data, columns, C, l, phi,f):
    newC = []

    for x in data[columns].values:
        if x not in C[0]:
            px = (l*distance(x, C[0])**2)/phi
            if px >= np.random.random(1):
                newC.append(x)

    if f == 0 :
        if len(newC) == 0:
            newC = C[0]
        else:
            newC = newC + C[0]

    return [newC, len(data)]

@task(returns=list)
def MergeBestMuKey(w1,w2):
    for i in range(len(w2)):
        w1[i]+=w2[i]
    return w1

@task(returns=list)
def bestMuKey(data,columns, C):
    C = C[0]
    w = [0 for i in xrange(len(C))]
    for x in np.array(data[columns]):
        distances = np.array([np.linalg.norm(x - np.array(c) ) for c in C])
        bestC = distances.argmin(axis=0)
        w[bestC]+=1

    return w
