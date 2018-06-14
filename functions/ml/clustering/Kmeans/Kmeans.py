#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import numpy as np
import pandas as pd
from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.functions.reduce import mergeReduce


class Kmeans(object):

    def fit(self, data, settings, nfrag):
        """
            fit():

            - :param data: A list with nfrag pandas's dataFrame
                   used to create the model.
            - :param settings: A dictionary that contains:
                - k: Number of wanted clusters.
                - features: Field of the features in the dataset;
                - maxIterations: Maximum number of iterations;
                - epsilon: Threshold to stop the iterations;
                - initMode: "random" or "k-means||"
            - :param nfrag: A number of fragments;
            - :return: Returns a model (which is a pandas dataFrame).
        """

        k = int(settings.get('k', 2))
        max_iterations = int(settings.get('maxIterations', 100))
        epsilon = float(settings.get('epsilon', 0.001))
        init_mode = settings.get('initMode', 'k-means||')

        if init_mode not in ['random','k-means||']:
            raise Exception("This implementation only suports 'random' and "
                            "k-means|| as a initialization mode. ")

        if 'features' not in settings:
           raise Exception("You must inform the `features` field.")

        features_col = settings['features']

        from pycompss.api.api import compss_wait_on
        # counting rows in each fragment
        size = [[] for _ in range(nfrag)]
        xp = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            xp[f] = _getSize(data[f], features_col, size[f])
        size = mergeReduce(_mergeCentroids, size)
        size = compss_wait_on(size)
        size, n_list = size

        if init_mode == "random":
            centroids = init_random(xp, k, size, n_list, nfrag)

        elif init_mode == "k-means||":
            centroids = init_parallel(xp, k, size, n_list, nfrag)
        else:
            raise Exception("Inform a valid initMode.")

        old_centroids = []
        it = 0

        while not has_converged(centroids, old_centroids,
                                epsilon, it, max_iterations):
            old_centroids = list(centroids)
            idx = [_find_closest_centroids(xp[f], centroids)
                   for f in range(nfrag)]

            idx = mergeReduce(_reduceCentersTask, idx)
            idx = compss_wait_on(idx)

            centroids = compute_centroids(idx)
            it += 1
            print '[INFO] - Iteration:' + str(it)

        model = dict()
        model['algorithm'] = 'K-Means'
        model['model'] = pd.DataFrame([[c] for c in centroids],
                                      columns=["Clusters"])

        return model

    def transform(self, data, model, settings, nfrag):
        """
            transform():

            :param data: A list with nfrag pandas's dataFrame
                    that will be predicted.
            :param model: The Kmeans model created;
            :param settings: A dictionary that contains:
                - features: Field of the features in the test data;
                - predCol: Alias to the new predicted labels;
            :param nfrag: A number of fragments;
            :return: The prediction (in the same input format).
        """

        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        if model.get('algorithm','null') != 'K-Means':
            raise Exception("You must inform a valid model.")

        model = model['model']
        features_col = settings['features']
        alias = settings.get('predCol', 'Prediction')

        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _assigment_cluster(data[f], features_col, model, alias)

        return result

    def transform_serial(self, data, model, settings):
        """
            transform():

            :param data: A list with nfrag pandas's dataFrame
                    that will be predicted.
            :param model: The Kmeans model created;
            :param settings: A dictionary that contains:
                - features: Field of the features in the test data;
                - predCol: Alias to the new predicted labels;
            :param nfrag: A number of fragments;
            :return: The prediction (in the same input format).
        """

        if 'features' not in settings:
            raise Exception("You must inform the `features` field.")

        if model.get('algorithm','null') != 'K-Means':
            raise Exception("You must inform a valid model.")

        model = model['model']
        features_col = settings['features']
        alias = settings.get('predCol', 'Prediction')

        result = _assigment_cluster_(data, features_col, model, alias)

        return result


@task(returns=list, size=INOUT)
def _getSize(data, columns, size):
    n = len(data)
    size += [n, [n]]
    XP = np.array(data[columns].values.tolist())
    return XP


@task(returns=list)
def _mergeCentroids(a, b):
    return [a[0] + b[0], a[1] + b[1]]


@task(returns=dict)
def _find_closest_centroids(XP, mu):
    """Find the closest centroid of each point in XP"""
    new_centroids = dict()
    k = len(mu)

    for i in range(k):
        new_centroids[i] = [0, []]

    for x in XP:
        distances = \
        np.array([np.linalg.norm(x - np.array(mu[j])) for j in range(k)])
        bestC = distances.argmin(axis=0)
        new_centroids[bestC][0] += 1
        new_centroids[bestC][1].append(x)

    for i in range(k):
        new_centroids[i][1] = np.sum(new_centroids[i][1], axis=0)
    return new_centroids


@task(returns=dict, priority=True)
def _reduceCentersTask(a, b):
    for key in b:
        if key not in a:
            a[key] = b[key]
        else:
            a[key] = (a[key][0] + b[key][0], a[key][1] + b[key][1])
    return a


def _compute_centroids(centroids):
    """ Next we need a function to compute the centroid of a cluster.
        The centroid is simply the mean of all of the examples currently
        assigned to the cluster."""
    def safe_div(x,y):
        if y == 0: return 0
        return x/y
    centroids = [safe_div(centroids[c][1], centroids[c][0]) for c in centroids]
    return centroids


@task(returns=list)
def _assigment_cluster(data, columns, model, alias):
    return _assigment_cluster_(data, columns, model, alias)


def _assigment_cluster_(data, columns, model, alias):
    mu = model['Clusters'].tolist()
    XP = np.array(data[columns].values)
    k = len(mu)
    values = []
    for x in XP:
        distances = np.array([np.linalg.norm(x - mu[j]) for j in range(k)])
        best_cluster = distances.argmin(axis=0)
        values.append(best_cluster)

    data[alias] = values
    return data


def has_converged(mu, oldmu, epsilon, iter, maxIterations):
    if len(oldmu) > 0:
        if iter < maxIterations:
            aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
            distancia = sum(aux)
            if distancia < (epsilon**2):
                return True
            else:
                return False
        else:
            return True


def distance(x, clusters):
    dist = min(np.array([np.linalg.norm(x - np.array(c)) for c in clusters]))
    return dist


# ------ INIT MODE: random
def init_random(xp, k, size, n_list, nfrag):
    from pycompss.api.api import compss_wait_on

    # define where get the inital core.
    # ids = np.sort(np.random.choice(size, k, replace=False))
    ids = _get_idx(size, k)
    list_ids = [[] for _ in range(nfrag)]

    frag = 0
    maxIdFrag = n_list[frag]
    oldmax = 0
    for i in ids:
        while i >= maxIdFrag:
            frag += 1
            oldmax = maxIdFrag
            maxIdFrag += n_list[frag]

        list_ids[frag].append(i-oldmax)

    fs_valids = [f for f in range(nfrag) if len(list_ids[f]) > 0]

    centroids = [_initMode_random(xp[f], list_ids[f]) for f in fs_valids]
    centroids = mergeReduce(_merge_initMode, centroids)
    centroids = compss_wait_on(centroids)
    centroids = centroids[0]
    return centroids


def _get_idx(size, k):
    n = []
    for i in range(k):
        n.append(np.random.random_integers(0, size))
    n = np.sort(n)
    return n


@task(returns=list)
def _initMode_random(xp, idxs):
    sampled = xp[idxs]
    return [sampled, 0]


@task(returns=list)
def _merge_initMode(a, b):
    if len(a[0]) == 0:
        return b
    elif len(b[0]) == 0:
        return a
    else:
        return [np.vstack((a[0], b[0])).tolist(), 0]


# ------ INIT MODE: k-means||
def init_parallel(xp, k, size, n_list, nfrag):
    """
    Initialize a set of cluster centers using the k-means|| algorithm
    by Bahmani et al. (Bahmani et al., Scalable K-Means++, VLDB 2012).
    This is a variant of k-means++ that tries to find dissimilar cluster
    centers by starting with a random center and then doing passes where
    more centers are chosen with probability proportional to their squared
    distance to the current cluster set. It results in a provable
    approximation to an optimal clustering.
    """
    from pycompss.api.api import compss_wait_on

    """
    Step1: C ← sample a point uniformly at random from X
    """
    fs_valids = [f for f in range(nfrag) if n_list[f] > 0]
    f = np.random.choice(fs_valids, 1, replace=False)[0]
    centroids = _initC(xp[f])

    for i in range(1, k):
        """
        Step2:  Compute 'cost' as the sum of all distances from xi to the
                closest point in centroids
        """
        dists = [_cost(xp[f], centroids) for f in range(nfrag)]
        info = mergeReduce(_merge_cost, dists)
        idx = _generate_candidate(info, n_list)
        idx = compss_wait_on(idx)
        centroids = _get_new_centroid(centroids, xp[f], idx)

    centroids = compss_wait_on(centroids)

    return centroids


@task(returns=list)
def _generate_candidate(info, n_list):

    distribution = info[0]/info[1]
    # Calculate the distribution for sampling a new center
    idx = np.random.choice(range(len(distribution)), 1, p=distribution)[0]
    frag = 0
    maxIdFrag = n_list[frag]
    oldmax = 0
    while idx >= maxIdFrag:
        frag += 1
        oldmax = maxIdFrag
        maxIdFrag += n_list[frag]
    idx = idx-oldmax
    return [idx]


@task(returns=list)
def _initC(xp):
    indices = np.random.randint(0,len(xp),1)
    sample = xp[indices]
    return sample


@task(returns=list)
def _get_new_centroid(centroids, xp, idx):
    idx = idx[0]
    x = xp[idx]
    centroids = np.concatenate((centroids,[x]))
    return centroids


@task(returns=list)
def _cost(XP, clusters):
    """Calculate the cost of data with respect to the current centroids."""

    dists = []
    tmp_costs = []
    for x in XP:
        dists.append(distance(x, clusters))

    return [dists, sum(dists)]


@task(returns=list)
def _merge_cost(info1, info2):
    dist = np.concatenate((info1[0],info2[0]))
    cost = info1[1] + info2[1]
    return [dist, cost]
