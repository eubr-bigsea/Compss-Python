#!/usr/bin/python
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


import numpy as np
import pandas as pd
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.local import *
from ddf.ddf import DDF
from ddf.ddf_model import ModelDDF

__all__ = ['Kmeans', 'DBSCAN']


import uuid
import sys
sys.path.append('../../')


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

    def __init__(self, feature_col, pred_col=None, n_clusters=3,
                 max_iters=100, epsilon=0.01, init_mode='k-means||'):
        """
        :param feature_col: Feature column name;
        :param pred_col: Output prediction column;
        :param n_clusters: Number of clusters;
        :param max_iters: Number maximum of iterations;
        :param epsilon: tolerance value (default, 0.01);
        :param init_mode: *'random'* or *'k-means||'*.
        """
        super(Kmeans, self).__init__()

        if not pred_col:
            pred_col = 'prediction_kmeans'

        if init_mode not in ['random', 'k-means||']:
            raise Exception("This implementation only suports 'random' and "
                            "k-means|| as a initialization mode. ")

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['pred_col'] = pred_col
        self.settings['max_iters'] = max_iters
        self.settings['init_mode'] = init_mode
        self.settings['k'] = n_clusters
        self.settings['epsilon'] = epsilon

        self.model = []
        self.name = 'Kmeans'
        self.cost = None

    def fit(self, data):
        """
        :param data: DDF
        :return: trained model
        """

        df, nfrag, tmp = self._ddf_inital_setup(data)

        # print compss_wait_on(df)

        k = int(self.settings['k'])
        max_iterations = int(self.settings.get('max_iters', 100))
        epsilon = float(self.settings.get('epsilon', 0.001))
        init_mode = self.settings.get('init_mode', 'k-means||')
        features_col = self.settings['feature_col']

        # counting rows in each fragment
        size = [[] for _ in range(nfrag)]
        xp = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            xp[f], size[f] = _kmeans_get_size(df[f], features_col)
        info = merge_reduce(_kmeans_mergeCentroids, size)
        info = compss_wait_on(info)
        size, n_list = info

        if init_mode == "random":
            centroids = _kmeans_init_random(xp, k, size, n_list, nfrag)

        elif init_mode == "k-means||":
            centroids = _kmeans_init_parallel(xp, k, size, n_list, nfrag)
        else:
            raise Exception("Inform a valid initMode.")

        old_centroids = []
        it = 0

        while not self._kmeans_has_converged(centroids, old_centroids,
                                             epsilon, it, max_iterations):
            old_centroids = list(centroids)
            idx = [_kmeans_find_closest_centroids(xp[f], centroids)
                   for f in range(nfrag)]
            idx = merge_reduce(_kmeans_reduceCentersTask, idx)
            centroids = _kmeans_compute_centroids(idx)
            it += 1
            print '[INFO] - Iteration:' + str(it)

        self.model = pd.DataFrame([[c] for c in centroids],
                                  columns=["Clusters"])
        return self

    def fit_transform(self, data):
        """
        Fit the model and transform.

        :param data: DDF
        :return: DDF
        """

        self.fit(data)
        ddf = self.transform(data)

        return ddf

    def transform(self, data):
        """

        :param data: DDF
        :return: trained model
        """

        if len(self.model) == 0:
            raise Exception("Model is not fitted.")

        df, nfrag, tmp = self._ddf_inital_setup(data)

        features_col = self.settings['feature_col']
        alias = self.settings['pred_col']

        result = [[] for _ in range(nfrag)]
        info = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _kmeans_assigment_cluster(df[f],
                                                           features_col,
                                                           self.model, alias)

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

    def _kmeans_has_converged(self, mu, oldmu, epsilon, iter, maxIterations):
        if len(oldmu) > 0:
            if iter < maxIterations:
                aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
                self.cost = sum(aux)
                if self.cost < (epsilon ** 2):
                    return True
                else:
                    return False
            else:
                return True


@task(returns=2)
def _kmeans_get_size(data, columns):
    n = len(data)
    size = [n, [n]]
    XP = np.array(data[columns].values.tolist())
    return XP, size


@task(returns=1)
def _kmeans_mergeCentroids(a, b):
    return [a[0] + b[0], a[1] + b[1]]


@task(returns=1)
def _kmeans_find_closest_centroids(XP, mu):
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


@task(returns=1)
def _kmeans_reduceCentersTask(a, b):
    for key in b:
        if key not in a:
            a[key] = b[key]
        else:
            a[key] = (a[key][0] + b[key][0], a[key][1] + b[key][1])
    return a

@local
def _kmeans_compute_centroids(centroids):
    """ Next we need a function to compute the centroid of a cluster.
        The centroid is simply the mean of all of the examples currently
        assigned to the cluster."""
    centroids = [np.divide(centroids[c][1], centroids[c][0]) for c in centroids]
    return centroids


@task(returns=2)
def _kmeans_assigment_cluster(data, columns, model, alias):

    mu = model['Clusters'].tolist()
    XP = np.array(data[columns].values)
    k = len(mu)
    values = []
    for x in XP:
        distances = np.array([np.linalg.norm(x - mu[j]) for j in range(k)])
        best_cluster = distances.argmin(axis=0)
        values.append(best_cluster)

    data[alias] = values
    
    info = [data.columns.tolist(), data.dtypes.values, [len(data)]]
    return data, info


def _kmeans_distance(x, clusters):
    dist = min(np.array([np.linalg.norm(x - np.array(c)) for c in clusters]))
    return dist


# ------ INIT MODE: random
def _kmeans_init_random(xp, k, size, n_list, nfrag):

    # define where get the inital core.
    # ids = np.sort(np.random.choice(size, k, replace=False))
    ids = _kmeans_get_idx(size, k)
    list_ids = [[] for _ in range(nfrag)]

    acc = 0
    acc_old = 0
    for nfrag, limit in enumerate(n_list):
        acc += limit
        ns = [i for i in ids if i < acc]
        list_ids[nfrag] = np.subtract(ns, acc_old).tolist()
        acc_old = acc
        ids = ids[len(ns):]
        if len(ids) == 0:
            break

    fs_valids = [f for f, values in enumerate(list_ids) if len(values) > 0]

    centroids = [_kmeans_initMode_random(xp[f], list_ids[f]) for f in fs_valids]
    centroids = merge_reduce(_kmeans_merge_initMode, centroids)
    centroids = compss_wait_on(centroids)

    centroids = centroids[0]
    return centroids


def _kmeans_get_idx(size, k):
    import random
    n = random.sample(range(0, size), k)
    n = np.sort(n)
    return n


@task(returns=list)
def _kmeans_initMode_random(xp, idxs):
    sampled = xp[idxs]
    return [sampled, 0]


@task(returns=list)
def _kmeans_merge_initMode(a, b):
    if len(a[0]) == 0:
        return b
    elif len(b[0]) == 0:
        return a
    else:
        return [np.vstack((a[0], b[0])).tolist(), 0]


# ------ INIT MODE: k-means||
def _kmeans_init_parallel(xp, k, size, n_list, nfrag):
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

    fs_valids = [f for f, values in enumerate(n_list) if values > 0]
    f = np.random.choice(fs_valids, 1, replace=False)[0]
    centroids = _kmeans_initC(xp[f])

    for i in range(1, k):
        """
        Step2:  Compute 'cost' as the sum of all distances from xi to the
                closest point in centroids
        """
        dists = [_kmeans_cost(xp[f], centroids) for f in range(nfrag)]
        info = merge_reduce(_kmeans_merge_cost, dists)
        idx, f = _kmeans_generate_candidate(info, n_list)
        centroids = _kmeans_get_new_centroid(centroids, xp[f], idx)

    centroids = compss_wait_on(centroids)

    print "centroids", centroids

    return centroids

@local
def _kmeans_generate_candidate(info, n_list):

    distribution = info[0]/info[1]
    # Calculate the distribution for sampling a new center
    idx = np.random.choice(range(len(distribution)), 1, p=distribution)[0]

    acc = 0
    for nfrag, limit in enumerate(n_list):
        if idx < limit:
            print [idx, nfrag]
            return [idx, nfrag]
        idx -= limit

    return [idx, len(n_list)]


@task(returns=list)
def _kmeans_initC(xp):
    indices = np.random.randint(0, len(xp), 1)
    sample = xp[indices]
    return sample


@task(returns=list)
def _kmeans_get_new_centroid(centroids, xp, idx):

    x = xp[idx]
    centroids = np.concatenate((centroids, [x]))
    return centroids


@task(returns=list)
def _kmeans_cost(XP, clusters):
    """Calculate the cost of data with respect to the current centroids."""

    dists = []
    for x in XP:
        dists.append(_kmeans_distance(x, clusters))

    return [dists, sum(dists)]


@task(returns=list)
def _kmeans_merge_cost(info1, info2):
    dist = np.concatenate((info1[0], info2[0]))
    cost = info1[1] + info2[1]
    return [dist, cost]


class DBSCAN(object):
    """
    Density-based spatial clustering of applications with noise (DBSCAN) is
    a data clustering algorithm.  It is a density-based clustering algorithm:
    given a set of points in some space, it groups together points that are
    closely packed together (points with many nearby neighbors), marking as
    outliers points that lie alone in low-density regions (whose nearest
    neighbors are too far away).

    .. warning:: Unstable

    :Example:

    >>> ddf2 = DBSCAN(feature_col='features', eps=0.01,
    >>>               min_pts=15).fit_predict(ddf1)
    """

    def __init__(self, feature_col, pred_col=None, eps=0.1, min_pts=15):
        """
        :param feature_col: Feature column name;
        :param pred_col: Output predicted column;
        :param eps: Epsilon distance (default is 0.1);
        :param min_pts: Minimum number of points to consider as a cluster.

        .. note:: Features need to be normalized.
        """

        if not pred_col:
            pred_col = 'prediction_DBSCAN'

        self.settings = dict()
        self.settings['feature_col'] = feature_col
        self.settings['pred_col'] = pred_col
        self.settings['eps'] = eps
        self.settings['min_pts'] = min_pts

        self.model = []
        self.name = 'DBSCAN'

    def fit_predict(self, data):
        """
        Fit and predict.

        :param data: DDF
        :return: DDF
        """

        df = data.partitions[0]
        nfrag = len(df)

        eps = self.settings['eps']

        div = int(np.sqrt(nfrag))
        grids = _fragment(div, eps)
        nlat = nlon = div

        # stage1 and stage2: _partitionize and local dbscan
        t = 0
        partial = [[] for _ in range(nfrag)]
        for l in range(div):
            for c in range(div):
                frag = []
                for f in range(nfrag):
                    frag = _partitionize(df[f], self.settings, grids[t], frag)

                partial[t] = _partial_dbscan(frag,
                                             self.settings, "p_{}_".format(t))
                t += 1

        # stage3: combining clusters
        n_iters_diagonal = (nlat - 1) * (nlon - 1) * 2
        n_iters_horizontal = (nlat - 1) * nlon
        n_iters_vertial = (nlon - 1) * nlat
        n_inters_total = n_iters_vertial + n_iters_horizontal + n_iters_diagonal
        mapper = [[] for _ in range(n_inters_total)]
        m = 0
        for t in range(nlat * nlon):
            i = t % nlon  # column
            j = t / nlon  # line
            if i < (nlon - 1):  # horizontal
                mapper[m] = _combine_clusters(partial[t], partial[t + 1])
                m += 1

            if j < (nlat - 1):
                mapper[m] = _combine_clusters(partial[t], partial[t + nlon])
                m += 1

            if i < (nlon - 1) and j < (nlat - 1):
                mapper[m] = _combine_clusters(partial[t], partial[t + nlon + 1])
                m += 1

            if i > 0 and j < (nlat - 1):
                mapper[m] = _combine_clusters(partial[t], partial[t + nlon - 1])
                m += 1

        merged_mapper = merge_reduce(_merge_mapper, mapper)
        components = _find_components(merged_mapper)

        # stage4: update_clusters
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _update_clusters(partial[f], components, grids[f])

        uuid_key = self._ddf_add_task(task_name='task_transform_dbscan',
                                      status='COMPLETED', lazy=False,
                                      function={0: result},
                                      parent=[data.last_uuid],
                                      n_output=1, n_input=1)

        data._set_n_input(uuid_key, data.settings['input'])
        return DDF(task_list=data.task_list, last_uuid=uuid_key)


# -----------------------------------------------------------------------------
#
#       stage1 and stage2: _partitionize in 2dim and local dbscan
#
# ------------------------------------------------------------------------------


def _fragment(div, eps):
    """Create a list of grids."""
    grids = []
    for lat in range(div):
        for log in range(div):
            init = [(1.0 / div) * lat, (1.0 / div) * log]
            end = [(1.0 / div) * (lat + 1) + 2 * eps,
                   (1.0 / div) * (log + 1) + 2 * eps]
            end2 = [(1.0 / div) * (lat + 1), (1.0 / div) * (log + 1)]
            grids.append([init, end, end2])
    return grids


def _inblock(row, column, init, end):
    """Check if point is in grid."""
    return all([row[column][0] >= init[0],
                row[column][1] >= init[1],
                row[column][0] <= end[0],
                row[column][1] <= end[1]])


@task(returns=list)
def _partitionize(df, settings, grids, frag):
    """Select points that belongs to each grid."""
    column = settings['feature']
    if len(df) > 0:
        init, end, end2 = grids
        tmp = df.apply(lambda row: _inblock(row, column, init, end), axis=1)
        tmp = df.loc[tmp]

        if len(frag) > 0:
            frag = pd.concat([frag, tmp])
        else:
            frag = tmp
    return frag


@task(returns=list)
def _partial_dbscan(df, settings, sufix):
    """Perform a partial dbscan."""
    stack = []
    cluster_label = 0
    UNMARKED = -1

    df = df.reset_index(drop=True)
    num_ids = len(df)
    eps = settings['eps']
    minPts = settings['minPts']
    columns = settings['feature']
    cluster_col = settings['predCol']

    # creating a new tmp_id
    cols = df.columns
    id_col = 'tmp_dbscan'
    i = 0
    while id_col in cols:
        id_col = 'tmp_dbscan_{}'.format(i)
        i += 1
    df[id_col] = ['id{}{}'.format(sufix, j) for j in range(len(df))]
    settings['idCol'] = id_col

    C_UNMARKED = "{}{}".format(sufix, UNMARKED)
    C_NOISE = "{}{}".format(sufix, '-0')
    df[cluster_col] = [C_UNMARKED for _ in range(num_ids)]

    df = df.reset_index(drop=True)

    for index in range(num_ids):
        point = df.loc[index]
        CLUSTER = point[cluster_col]

        if CLUSTER == C_UNMARKED:
            X = _retrieve_neighbors(df, index, point, eps, columns)

            if len(X) < minPts:
                df.loc[df.index[index], cluster_col] = C_NOISE
            else:  # found a core point and assign a label to this point
                cluster_label += 1
                df.loc[index, cluster_col] = sufix + str(cluster_label)
                for new_index in X:  # assign core's label to its neighborhood
                    label = sufix + str(cluster_label)
                    df.loc[df.index[new_index], cluster_col] = label
                    if new_index not in stack:
                        stack.append(new_index)  # append neighborhood to stack
                    while len(stack) > 0:
                        # find new neighbors from core point neighborhood
                        newest_index = stack.pop()
                        new_point = df.loc[newest_index]
                        Y = _retrieve_neighbors(df, newest_index,
                                                new_point, eps, columns)

                        if len(Y) >= minPts:
                            # current_point is a new core
                            for new_index_neig in Y:
                                neig_cluster = \
                                    df.loc[new_index_neig][cluster_col]
                                if neig_cluster == C_UNMARKED:
                                    df.loc[df.index[new_index_neig],
                                           cluster_col] = \
                                        sufix + str(cluster_label)
                                    if new_index_neig not in stack:
                                        stack.append(new_index_neig)

    settings['clusters'] = df[cluster_col].unique()
    return [df, settings]


def _retrieve_neighbors(df, i_point, point, eps, column):
    """Retrieve the list of neigbors."""
    neigborhood = []
    for index, row in df.iterrows():
        if index != i_point:
            a = np.array(point[column])
            b = np.array([row[column]])
            distance = np.linalg.norm(a - b)
            if distance <= eps:
                neigborhood.append(index)

    return neigborhood


# -----------------------------------------------------------------------------
#
#                   #stage3: combining clusters
#
# -----------------------------------------------------------------------------


@task(returns=dict)
def _combine_clusters(p1, p2):
    """Identify which points are duplicated in each grid pair."""
    df1, settings = p1
    df2 = p2[0]
    primary_key = settings['idCol']
    cluster_col = settings['predCol']
    a = settings['clusters']
    b = p2[1]['clusters']
    unique_c = np.unique(np.concatenate((a, b), 0))
    links = []

    if len(df1) > 0 and len(df2) > 0:
        merged = pd.merge(df1, df2, how='inner', on=[primary_key])

        for index, point in merged.iterrows():
            CLUSTER_DF1 = point[cluster_col + "_x"]
            CLUSTER_DF2 = point[cluster_col + "_y"]
            link = [CLUSTER_DF1, CLUSTER_DF2]
            if link not in links:
                links.append(link)

    result = dict()
    result['cluster'] = unique_c
    result['links'] = links
    return result


@task(returns=dict)
def _merge_mapper(mapper1, mapper2):
    """Merge all the duplicated points list."""
    if len(mapper1) > 0:
        if len(mapper2) > 0:
            clusters1 = mapper1['cluster']
            clusters2 = mapper2['cluster']
            clusters = np.unique(np.concatenate((clusters1, clusters2), 0))

            mapper1['cluster'] = clusters
            mapper1['links'] += mapper2['links']
    else:
        mapper1 = mapper2
    return mapper1


@task(returns=dict)
def _find_components(merged_mapper):
    """Find the list of components presents in the duplicated points."""
    import networkx as nx

    AllClusters = merged_mapper['cluster']  # list of unique clusters
    links = merged_mapper['links']  # links represents the same point
    i = 0

    G = nx.Graph()
    for line in links:
        G.add_edge(line[0], line[1])

    components = sorted(nx.connected_components(G), key=len, reverse=True)

    oldC_newC = dict()  # from: 'old id cluster' to: 'new id cluster'

    for component in components:
        hascluster = any(["-0" not in c for c in component])
        if hascluster:
            i += 1
            for c in component:
                if "-0" not in c:
                    oldC_newC[c] = i

    # rename others clusters
    for c in AllClusters:
        if c not in oldC_newC:
            if "-0" not in c:
                i += 1
                oldC_newC[c] = i

    return oldC_newC


# -----------------------------------------------------------------------------
#
#                   #stage4: Update Clusters
#
# ------------------------------------------------------------------------------


@task(returns=list)
def _update_clusters(partial, to_update, grids):
    """Update the information about the cluster."""
    df, settings = partial
    clusters = settings['clusters']
    primary_key = settings['idCol']
    cluster_id = settings['predCol']
    column = settings['feature']
    init, end, end2 = grids

    if len(df) > 0:
        tmp = df.apply(lambda row: _inblock(row, column, init, end2), axis=1)
        df = df.loc[tmp]
        df.drop_duplicates([primary_key], inplace=False)
        df = df.reset_index(drop=True)

        for key in to_update:
            if key in clusters:
                df.loc[df[cluster_id] == key, cluster_id] = to_update[key]

        df.loc[df[cluster_id].str.contains("-0", na=False), cluster_id] = -1
        df = df.drop([primary_key], axis=1)

    return df
