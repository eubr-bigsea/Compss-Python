#!/usr/bin/python
# -*- coding: utf-8 -*-
"""DBSCAN.

Density-based spatial clustering of applications with noise (DBSCAN) is
a data clustering algorithm.  It is a density-based clustering algorithm:
given a set of points in some space, it groups together points that are
closely packed together (points with many nearby neighbors), marking as
outliers points that lie alone in low-density regions (whose nearest
neighbors are too far away).
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import pandas as pd
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce


class DBSCAN(object):
    """DBSCAN's methods.

    - fit_predict(): Perform DBSCAN clustering from the features column.
    """

    def fit_predict(self, df, settings, numFrag):
        """fit_predict.

        :param df:       A list with numFrag pandas's dataframe.
        :param settings: A dictionary that contains:
            - feature:   Field  of the normalizated features in the test data;
            - idCol:     Column name to be a primary key of the dataframe;
            - predCol:   Alias to the new column with the labels predicted;
            - minPts:    The number of samples in a neighborhood for a point
                         to be considered as a core point;
                         This includes the point itself. (int, default: 15)
            - eps:       The maximum distance between two samples for them to
                         be considered as in the same neighborhood.
                         (float, default: 0.1)
        :param numFrag:  Number of even fragments;
        :return:         Returns a list of dataframe with the cluster column.
        """
        minPts = settings.get('minPts', 15)
        eps = settings.get('eps',   0.1)
        predCol = settings.get('predCol', "Cluster")
        settings['eps'] = eps
        settings['minPts'] = minPts
        settings['predCol'] = predCol
        if not all(['feature' in settings,
                    'idCol' in settings]):
            raise Exception("Please inform, at least, the fields: "
                            "`idCol`,`feature`")

        div = int(np.sqrt(numFrag))
        grids = fragment(div, eps)
        nlat = nlon = div
        # stage1 and stage2: partitionize and local dbscan
        t = 0
        partial = [[] for f in range(numFrag)]
        for l in range(div):
            for c in range(div):
                frag = []
                for f in range(numFrag):
                    frag = partitionize(df[f], settings, grids[t], frag)

                partial[t] = partial_dbscan(frag, settings, "p_{}_".format(t))
                t += 1

        # stage3: combining clusters
        n_iters_diagonal = (nlat-1)*(nlon-1)*2
        n_iters_horizontal = (nlat-1)*nlon
        n_iters_vertial = (nlon-1)*nlat
        n_inters_total = n_iters_vertial+n_iters_horizontal+n_iters_diagonal
        mapper = [[] for f in range(n_inters_total)]
        m = 0
        for t in range(nlat*nlon):
            i = t % nlon  # column
            j = t / nlon  # line
            if i < (nlon-1):  # horizontal
                mapper[m] = CombineClusters(partial[t], partial[t+1])
                m += 1

            if j < (nlat-1):
                mapper[m] = CombineClusters(partial[t], partial[t+nlon])
                m += 1

            if i < (nlon-1) and j < (nlat-1):
                mapper[m] = CombineClusters(partial[t], partial[t+nlon+1])
                m += 1

            if i > 0 and j < (nlat-1):
                mapper[m] = CombineClusters(partial[t], partial[t+nlon-1])
                m += 1

        merged_mapper = mergeReduce(MergeMapper, mapper)
        components = findComponents(merged_mapper, minPts)

        # stage4: updateClusters
        t = 0
        result = [[] for f in range(numFrag)]
        for t in range(numFrag):
            result[t] = updateClusters(partial[t], components, grids[t])

        return result

# -----------------------------------------------------------------------------
#
#       stage1 and stage2: partitionize in 2dim and local dbscan
#
# ------------------------------------------------------------------------------


def fragment(div, eps):
    """Create a list of grids."""
    grids = []
    for lat in range(div):
        for log in range(div):
            init = [(1.0/div)*lat,  (1.0/div)*log]
            end = [(1.0/div)*(lat+1)+2*eps, (1.0/div)*(log+1)+2*eps]
            end2 = [(1.0/div)*(lat+1), (1.0/div)*(log+1)]
            grids.append([init, end, end2])
    return grids


def inblock(row, column, init, end):
    """Check if point is in grid."""
    return all([row[column][0] >= init[0],
                row[column][1] >= init[1],
                row[column][0] <= end[0],
                row[column][1] <= end[1]])


@task(returns=list)
def partitionize(df, settings, grids, frag):
    """Select points that belongs to each grid."""
    column = settings['feature']
    if len(df) > 0:
        init, end, end2 = grids
        f = lambda row: inblock(row, column, init, end)
        tmp = df.apply(f, axis=1)
        tmp = df.loc[tmp]

        if len(frag) > 0:
            frag = pd.concat([frag, tmp])
        else:
            frag = tmp
    return frag


@task(returns=list)
def partial_dbscan(df, settings, sufix):
    """Perform a partial dbscan."""
    stack = []
    cluster_label = 0
    UNMARKED = -1

    df = df.reset_index(drop=True)
    num_ids = len(df)
    eps = settings['eps']
    minPts = settings['minPts']
    columns = settings['feature']
    clusterCol = settings['predCol']
    # idCol = settings['idCol']

    C_UNMARKED = "{}{}".format(sufix, UNMARKED)
    C_NOISE = "{}{}".format(sufix, '-0')
    df[clusterCol] = [C_UNMARKED for i in range(num_ids)]

    for index in range(num_ids):
        point = df.loc[index]
        CLUSTER = point[clusterCol]

        if CLUSTER == C_UNMARKED:
            X = retrieve_neighbors(df, index, point, eps, columns)

            if len(X) < minPts:
                df.set_value(index, clusterCol, C_NOISE)
            else:   # found a core point
                cluster_label += 1
                # assign a label to core point
                df.set_value(index, clusterCol, sufix + str(cluster_label))
                for new_index in X:  # assign core's label to its neighborhood
                    label = sufix + str(cluster_label)
                    df.set_value(new_index, clusterCol, label)
                    if new_index not in stack:
                        stack.append(new_index)  # append neighborhood to stack
                    while len(stack) > 0:
                        # find new neighbors from core point neighborhood
                        newest_index = stack.pop()
                        new_point = df.loc[newest_index]
                        Y = retrieve_neighbors(df, newest_index,
                                               new_point, eps, columns)

                        if len(Y) >= minPts:
                            # current_point is a new core
                            for new_index_neig in Y:
                                neig_cluster = \
                                    df.loc[new_index_neig][clusterCol]
                                if (neig_cluster == C_UNMARKED):
                                    label = sufix + str(cluster_label)
                                    df.set_value(new_index_neig,
                                                 clusterCol,
                                                 label)
                                    if new_index_neig not in stack:
                                        stack.append(new_index_neig)

    settings['clusters'] = df[clusterCol].unique()
    return [df, settings]


def retrieve_neighbors(df, i_point, point, eps, column):
    """Retrieve the list of neigbors."""
    neigborhood = []
    for index, row in df.iterrows():
        if index != i_point:
                a = np.array(point[column])
                b = np.array([row[column]])
                distance = np.linalg.norm(a-b)
                if distance <= eps:
                    neigborhood.append(index)

    return neigborhood

# -----------------------------------------------------------------------------
#
#                   #stage3: combining clusters
#
# -----------------------------------------------------------------------------


@task(returns=dict)
def CombineClusters(p1, p2):
    """Identify which points are duplicated in each grid pair."""
    df1, settings = p1
    df2 = p2[0]
    primary_key = settings['idCol']
    clusterCol = settings['predCol']
    a = settings['clusters']
    b = p2[1]['clusters']
    unique_c = np.unique(np.concatenate((a, b), 0))
    links = []

    if len(df1) > 0 and len(df2) > 0:
        merged = pd.merge(df1, df2, how='inner', on=[primary_key])

        for index, point in merged.iterrows():
            CLUSTER_DF1 = point[clusterCol+"_x"]
            CLUSTER_DF2 = point[clusterCol+"_y"]
            link = [CLUSTER_DF1, CLUSTER_DF2]
            if link not in links:
                links.append(link)

    result = dict()
    result['cluster'] = unique_c
    result['links'] = links
    return result


@task(returns=dict)
def MergeMapper(mapper1, mapper2):
    """Merge all the duplicated points list."""
    clusters1 = mapper1['cluster']
    clusters2 = mapper2['cluster']
    clusters = np.unique(np.concatenate((clusters1, clusters2), 0))

    mapper1['cluster'] = clusters
    mapper1['links'] += mapper2['links']
    return mapper1


@task(returns=dict)
def findComponents(merged_mapper, minPts):
    """Find the list of components presents in the duplicated points."""
    import networkx as nx

    AllClusters = merged_mapper['cluster']  # list of unique clusters
    links = merged_mapper['links']   # links represents the same point
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
#                   #stage4: updateClusters
#
# ------------------------------------------------------------------------------


@task(returns=list)
def updateClusters(partial, oldC_newC, grids):
    """Update the information about the cluster."""
    df1, settings = partial
    clusters = settings['clusters']
    primary_key = settings['idCol']
    clusterCol = settings['predCol']
    column = settings['feature']
    init, end, end2 = grids

    if len(df1) > 0:
        f = lambda row: inblock(row, column, init, end2)
        tmp = df1.apply(f, axis=1)
        df1 = df1.loc[tmp]
        df1.drop_duplicates([primary_key], inplace=False)
        df1 = df1.reset_index(drop=True)

        for key in oldC_newC:
            if key in clusters:
                df1.ix[df1[clusterCol] == key, clusterCol] = oldC_newC[key]

        df1.ix[df1[clusterCol].str.contains("-0", na=False), clusterCol] = -1

    return df1
