#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF, generate_info
from ddf_library.ddf_model import ModelDDF

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.functions.reduce import merge_reduce
# from pycompss.api.local import *  # it requires guppy

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


class Kmeans(ModelDDF):
    """
    K-means clustering is a type of unsupervised learning, which is used when
    you have unlabeled data (i.e., data without defined categories or groups).
    The goal of this algorithm is to find groups in the data, with the number of
    groups represented by the variable K. The algorithm works iteratively to
    assign each data point to one of K groups based on the features that are
    provided. Data points are clustered based on feature similarity.

    Two of the most well-known forms of initialization of the set of clusters
    are: "random" and "k-means||":

     * random: Starting with a set of randomly chosen initial centers;
     * k-means|| (Bahmani et al., Scalable K-Means++, VLDB 2012): This is a
       variant of k-means++ that tries to find dissimilar cluster centers by
       starting with a random center and then doing passes where more centers
       are chosen with probability proportional to their squared distance to the
       current cluster set. It results in a provable approximation to an optimal
       clustering.

    :Example:

    >>> kmeans = Kmeans(feature_col='features', n_clusters=2,
    >>>                 init_mode='random').fit(ddf1)
    >>> ddf2 = kmeans.transform(ddf1)
    """

    def __init__(self, feature_col, n_clusters=3, max_iters=100,
                 epsilon=0.01, init_mode='k-means||'):
        """
        :param feature_col: Feature column name;
        :param n_clusters: Number of clusters;
        :param max_iters: Number maximum of iterations;
        :param epsilon: tolerance value (default, 0.01);
        :param init_mode: *'random'* or *'k-means||'*.
        """
        super(Kmeans, self).__init__()

        if init_mode not in ['random', 'k-means||']:
            raise Exception("This implementation only suports 'random' and "
                            "k-means|| as a initialization mode. ")

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['max_iters'] = max_iters
        self.settings['init_mode'] = init_mode
        self.settings['k'] = n_clusters
        self.settings['epsilon'] = epsilon

        self.model = dict()
        self.name = 'Kmeans'
        self.cost = np.inf

    def fit(self, data):
        """
        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        k = int(self.settings['k'])
        max_iterations = int(self.settings.get('max_iters', 100))
        epsilon = float(self.settings.get('epsilon', 0.001))
        init_mode = self.settings.get('init_mode', 'k-means||')
        features_col = self.settings['feature_col']

        # counting rows in each fragment
        size = [[] for _ in range(nfrag)]
        xp = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            xp[f], size[f] = _kmeans_project_fields(df[f], features_col)
        info = merge_reduce(_kmeans_merge_centroids, size)
        info = compss_wait_on(info)
        size, n_list = info

        if size < k:
            raise Exception("Number of Clusters K is greather "
                            "than number of rows.")

        if init_mode == "random":
            centroids = _kmeans_init_random(xp, k, size, n_list)

        elif init_mode == "k-means||":
            centroids = _kmeans_init_parallel(xp, k, n_list, nfrag)
        else:
            raise Exception("Inform a valid initMode.")

        it = 0
        cost = np.inf

        while not self._kmeans_has_converged(cost, epsilon, it, max_iterations):
            idx = [_kmeans_find_closest(xp[f], centroids) for f in range(nfrag)]
            idx = merge_reduce(_kmeans_merge_keys, idx)
            centroids, cost = _kmeans_compute_centroids(idx)

            it += 1
            print('[INFO] - KMeans - it {} cost: {}'.format(it, cost))

        self.model = centroids
        return self

    def fit_transform(self, data, pred_col='prediction_kmeans'):
        """
        Fit the model and transform.

        :param data: DDF
        :param pred_col: Output prediction column;
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data, pred_col=pred_col)

        return ddf

    def transform(self, data, pred_col='prediction_kmeans'):
        """

        :param data: DDF
        :param pred_col: Output prediction column;
        :return: trained model
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        df, nfrag, tmp = self._ddf_inital_setup(data)

        features_col = self.settings['feature_col']
        self.settings['pred_col'] = pred_col

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _kmeans_predict(df[f], features_col,
                                                 self.model, pred_col, f)

        uuid_key = self._ddf_add_task(task_name='task_transform_kmeans',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        self._set_n_input(uuid_key, 0)
        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)

    def compute_cost(self):
        """
        Compute the cost of this iteration;

        :return: float
        """

        if self.cost is None:
            raise Exception("Model is not fitted.")
        else:
            return self.cost

    def _kmeans_has_converged(self, cost, epsilon, it, max_iter):
        """
        Terminate when residual sum of squares (RSS) falls below a threshold.

        :param epsilon:
        :param it:
        :param max_iter:
        :return:
        """

        old_cost = self.cost
        self.cost = cost
        if it < max_iter:
            if np.abs(self.cost - old_cost) < epsilon:
                return True
            else:
                return False
        else:
            return True


@task(returns=2)
def _kmeans_project_fields(data, columns):

    xp = data[columns].dropna().values
    n = len(xp)
    size = [n, [n]]
    return xp, size


@task(returns=1)
def _kmeans_merge_centroids(a, b):
    return [a[0] + b[0], a[1] + b[1]]


@task(returns=1)
def _kmeans_find_closest(xp, centroids):
    """Find the closest centroid of each point in XP"""

    n_clusters = len(centroids)
    """
    new_centroids = dict()
    key is a centroid = [ value1 is the number of points assigned
                          value2 is the sum of all these points
                          value3 is the rss]
    """

    new_centroids = dict()
    for k in range(n_clusters):
        new_centroids[k] = [0, [], []]

    if len(xp) > 0:

        distances = euclidean_distances(xp, centroids, squared=False)
        bests = distances.argmin(axis=1)

        for k in new_centroids:
            points = xp[bests == k]
            new_centroids[k][0] = len(points)
            new_centroids[k][1] = np.sum(points, axis=0)
            new_centroids[k][2] = np.sum(distances[bests == k])

    return new_centroids


@task(returns=1)
def _kmeans_merge_keys(a, b):
    for key in b:
        if key not in a:
            a[key] = b[key]
        else:
            a[key] = (a[key][0] + b[key][0],
                      a[key][1] + b[key][1],
                      a[key][2] + b[key][2])
    return a


# @local
def _kmeans_compute_centroids(centroids):
    """ Next we need a function to compute the centroid of a cluster.
        The centroid is simply the mean of all of the examples currently
        assigned to the cluster."""
    centroids = compss_wait_on(centroids)
    costs = np.sum([centroids[c][2] for c in centroids])
    centroids = [np.divide(centroids[c][1], centroids[c][0])
                 for c in centroids]
    return np.array(centroids), costs


@task(returns=2)
def _kmeans_predict(data, columns, model, alias, frag):
    centroids = model

    if len(data) > 0:
        k = len(centroids)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.cluster_centers_ = centroids

        values = kmeans.predict(data[columns].values)

    else:
        values = np.nan

    if alias in data.columns:
        data.drop([alias], axis=1, inplace=True)

    data[alias] = values

    info = generate_info(data, frag)
    return data, info


# ------ INIT MODE: random
def _kmeans_init_random(xp, k, size, n_list):
    # define where get the inital core.
    ids = _kmeans_select_random_seeds(size, k)

    from ddf_library.utils import divide_idx_in_frags
    list_ids = divide_idx_in_frags(ids, n_list)

    # it will run tasks only to selected partitions
    sel_frags = [f for f, values in enumerate(list_ids) if len(values) > 0]

    centroids = [_kmeans_get_random(xp[f], list_ids[f]) for f in sel_frags]
    centroids = merge_reduce(_kmeans_merge_randoms, centroids)
    centroids = compss_wait_on(centroids)

    return centroids


def _kmeans_select_random_seeds(size, k):
    idx = np.random.randint(0, size + 1, k)
    idx = np.unique(idx)
    while len(idx) != k:
        tmp = np.random.randint(0, size + 1, 1)
        if tmp[0] not in idx:
            idx = np.append(idx, tmp)

    return idx


@task(returns=1)
def _kmeans_get_random(xp, idxs):
    sampled = xp[idxs]
    return sampled


@task(returns=1)
def _kmeans_merge_randoms(a, b):
    if len(a) == 0:
        return b
    elif len(b) == 0:
        return a
    else:
        return np.vstack((a, b))


# ------ INIT MODE: k-means||
def _kmeans_init_parallel(xp, k, n_list, nfrag):
    """
    Initialize a set of cluster centers using the k-means|| algorithm
    by Bahmani et al. (Bahmani et al., Scalable K-Means++, VLDB 2012).
    This is a variant of k-means++ that tries to find dissimilar cluster
    centers by starting with a random center and then doing passes where
    more centers are chosen with probability proportional to their squared
    distance to the current cluster set. It results in a provable
    approximation to an optimal clustering.
    """

    """
    Step1: C ← sample a point uniformly at random from X
    """

    oversampling_l = 2  # oversampling factor

    fs_valids = [f for f, values in enumerate(n_list) if values > 0]
    f = np.random.choice(fs_valids, 1, replace=False)[0]
    centroids = _kmeans_init_clusters(xp[f])

    """
    Step2:  Compute 'cost' as the sum of all distances from xi to the
            closest point in centroids
    """
    dists = [_kmeans_cost(xp[f], centroids) for f in range(nfrag)]
    info = merge_reduce(_kmeans_merge_cost, dists)
    rss_n = compss_wait_on(info)
    phi = int(np.rint(np.log(rss_n[0])))
    print("[INFO] Starting K-means|| with log(phi):", phi)

    for _ in range(phi):
        candidates = [_kmeans_probality(xp[f], rss_n, centroids,
                                        oversampling_l, f)
                      for f in range(nfrag)]
        centroids = merge_reduce(_kmeans_merge_candidates, candidates)

        dists = [_kmeans_cost(xp[f], centroids) for f in range(nfrag)]
        rss_n = merge_reduce(_kmeans_merge_cost, dists)

    """
    To reduce the number of centers, assigns weights to the points in C 
    and reclusters these weighted points to obtain k centers
    """

    weights = [_kmeans_gen_weight(xp[f], centroids) for f in range(nfrag)]
    weights = merge_reduce(_kmeans_sum_weight, weights)

    weights = compss_wait_on(weights)
    weights = [k for k, y in weights.most_common()[:-k-1:-1]]

    centroids = compss_wait_on(centroids)
    centroids = centroids[weights]

    return centroids


@task(returns=1)
def _kmeans_init_clusters(xp):
    indices = np.random.randint(len(xp), size=1)
    sample = xp[indices, :]
    return sample


@task(returns=1)
def _kmeans_probality(xp, rss_n, centroids, l, frag):
    """
    px=l * d2(x,C)/ϕX(C)

    where:
     d2(x,C) is the distance to the closest center

     ϕX(C) is the sum of all smallest euclidean distance from all points
     from the set X to all points from  C

     The algorithm is simply:

        iterate in X to find all xi
        for each xi compute pxi
        generate an uniform number in [0,1], if is smaller than pxi select it to form C′
        after you done all draws include selected points from C′ into C
    :param xp:
    :param rss:
    :param centroids:
    :return:
    """
    rss, n = rss_n
    n_rows = len(xp)

    if n_rows > 0:

        distances = euclidean_distances(xp, centroids, squared=False)\
            .min(axis=1)
        px = (l * distances) / rss - np.random.random_sample(n_rows)

        idx = np.argwhere(px >= 0).flatten()
        xp = xp[idx]

    if frag == 0:
        if len(xp) > 0:
            xp = np.concatenate((centroids, xp))
        else:
            xp = centroids
    return xp


@task(returns=1)
def _kmeans_merge_candidates(c1, c2):

    if len(c1) > 0:
        if len(c2) > 0:
            return np.concatenate((c1, c2))
        else:
            return c1
    else:
        return c2


@task(returns=1)
def _kmeans_cost(xp, centroids):
    """Calculate the cost of data with respect to the current centroids."""
    n_rows = 0
    if len(xp) > 0:
        distances = euclidean_distances(xp, centroids, squared=False)\
            .min(axis=1)
        distances = np.sum(distances)
    else:
        distances = 0
    return [distances, n_rows]


@task(returns=1)
def _kmeans_merge_cost(info1, info2):
    cost = info1[0] + info2[0]
    n = info1[1] + info2[1]
    return [cost, n]


@task(returns=1)
def _kmeans_gen_weight(xp, centroids):
    from collections import Counter
    if len(xp) > 0:
        closest = euclidean_distances(xp, centroids, squared=False)\
            .argmin(axis=1)

        closest = Counter(closest)
        # same = Counter(np.arange(len(centroids)))
        closest = closest #- same
        return closest
    else:
        return Counter()


@task(returns=1)
def _kmeans_sum_weight(count1, count2):
    return count1 + count2