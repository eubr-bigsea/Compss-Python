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
from pycompss.functions.reduce import mergeReduce


class DBSCAN(object):
    """DBSCAN's methods.

    - fit_predict(): Perform DBSCAN clustering from the features column.
    """

    def fit_predict(self, df, settings, nfrag):
        """fit_predict.

        :param df: A list with nfrag pandas's dataframe.
        :param settings: A dictionary that contains:
            - feature: Field  of the normalizated features in the test data;
            - predCol: Alias to the new column with the labels predicted;
            - minPts: The number of samples in a neighborhood for a point
                to be considered as a core point;
                This includes the point itself. (int, default: 15)
            - eps: The maximum distance between two samples for them to 
                be considered as in the same neighborhood. (float, default: 0.1)
        :param nfrag: Number of even fragments;
        :return: Returns a list of dataframe with the cluster column.
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
                    frag = _partitionize(df[f], settings, grids[t], frag)

                partial[t] = _partial_dbscan(frag, settings, "p_{}_".format(t))
                t += 1

        # stage3: combining clusters
        n_iters_diagonal = (nlat-1)*(nlon-1)*2
        n_iters_horizontal = (nlat-1)*nlon
        n_iters_vertial = (nlon-1)*nlat
        n_inters_total = n_iters_vertial+n_iters_horizontal+n_iters_diagonal
        mapper = [[] for _ in range(n_inters_total)]
        m = 0
        for t in range(nlat*nlon):
            i = t % nlon  # column
            j = t / nlon  # line
            if i < (nlon-1):  # horizontal
                mapper[m] = _combine_clusters(partial[t], partial[t+1])
                m += 1

            if j < (nlat-1):
                mapper[m] = _combine_clusters(partial[t], partial[t+nlon])
                m += 1

            if i < (nlon-1) and j < (nlat-1):
                mapper[m] = _combine_clusters(partial[t], partial[t+nlon+1])
                m += 1

            if i > 0 and j < (nlat-1):
                mapper[m] = _combine_clusters(partial[t], partial[t+nlon-1])
                m += 1

        merged_mapper = mergeReduce(_merge_mapper, mapper)
        components = _find_components(merged_mapper)

        # stage4: update_clusters
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f] = _update_clusters(partial[f], components, grids[f])

        return result

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
            init = [(1.0/div)*lat,  (1.0/div)*log]
            end = [(1.0/div)*(lat+1)+2*eps, (1.0/div)*(log+1)+2*eps]
            end2 = [(1.0/div)*(lat+1), (1.0/div)*(log+1)]
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
            else:   # found a core point and assign a label to this point
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
                                           cluster_col] =\
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
            CLUSTER_DF1 = point[cluster_col+"_x"]
            CLUSTER_DF2 = point[cluster_col+"_y"]
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
