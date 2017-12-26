#!/usr/bin/python
# -*- coding: utf-8 -*-
"""PageRank.

PageRank is one of the methods Google uses to determine a page's
relevance or importance. The idea that Page Rank brought up was that, the
importance of any web page can be judged by looking at the pages that link
to it.

PageRank can be utilized in others domains. For example, may also be used
as a methodology to measure the apparent impact of a community.

See more at: http://www.cs.princeton.edu/~chazelle/courses/BIB/pagerank.htm
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import pandas as pd
from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.functions.reduce import mergeReduce


class PageRank(object):
    """PageRank's methods.

    - runPageRank(): Run PageRank for a fixed number of iterations returning a
    graph with vertex attributes containing the PageRank.
    """

    def runPageRank(self, data, settings, numFrag):
        """RunPageRank.

        :param data:        A list with numFrag pandas's dataframe.
        :param settings:    A dictionary that contains:
            * inlink:       Field of the inlinks vertex;
            * outlink:      Field of the outlinks vertex;
            * damping_factor: The coeficent of the  damping factor [0,1]
                            (default, 0.85);
            * maxIters:     The number of iterations (defaul, 100);
            * col1:         Alias of the vertex column (default, 'Vertex');
            * col2:         Alias of the ranking column (default, 'Rank').

        :param numFrag:     A number of fragments;
        :return:            A list of pandas's dataframe with the
                            ranking of each vertex in the dataset.
        """
        if 'inlink' not in settings or 'outlink' not in settings:
            raise Exception('Please inform at least')

        inlink = settings['inlink']
        outlink = settings['outlink']
        factor = settings.get('damping_factor', 0.85)
        maxIterations = settings.get('maxIters', 100)
        col1 = settings.get('col1', 'Vertex')
        col2 = settings.get('col2', 'Rank')

        adj, rank = create_AdjList(data, inlink, outlink, numFrag)

        for iteration in xrange(maxIterations):
            contributions = [calcContribuitions(adj[i], rank[i])
                             for i in range(numFrag)]
            merged_c = mergeReduce(mergeContribs, contributions)
            rank = [updateRank_p(rank[i], merged_c, factor)
                    for i in range(numFrag)]

        table = [printingResults(rank[i], col1, col2) for i in range(numFrag)]

        merged_table = mergeReduce(mergeRanks, table)
        result = split(merged_table, numFrag)
        return result


def create_AdjList(data, inlink, outlink, numFrag):
    """1o Load all URL's from the data and initialize their neighbors."""
    """2o Initialize each page’s rank to 1.0"""
    adjlist = [[] for i in range(numFrag)]
    rankslist = [[] for i in range(numFrag)]
    counts_in = [[] for i in range(numFrag)]

    for i in range(numFrag):
        adjlist[i] = partial_AdjList(data[i], inlink, outlink)
        rankslist[i] = partial_RankList(data[i], inlink, outlink)
        counts_in[i] = counts_inlinks(adjlist[i])

    counts_in = mergeReduce(merge_counts, counts_in)
    for i in range(numFrag):
        adjlist[i] = update_AdjList(adjlist[i], counts_in)

    return adjlist, rankslist


@task(returns=dict)
def partial_RankList(data, inlink, outlink):
    """Generate a partial rank list of each vertex."""
    ranks = {}
    cols = [outlink, inlink]
    for link in data[cols].values:
        v_out = link[0]
        v_in = link[1]
        if v_out not in ranks:
            ranks[v_out] = 1.0  # Rank, contributions, main
        if v_in not in ranks:
            ranks[v_in] = 1.0
    return ranks


@task(returns=dict)
def partial_AdjList(data, inlink, outlink):
    """Generate a partial adjacency list."""
    adj = {}
    for link in data[[outlink, inlink]].values:
        v_out = link[0]
        v_in = link[1]
        if v_out in adj:
            adj[v_out][0].append(v_in)
            adj[v_out][1] += 1
        else:
            adj[v_out] = [[v_in], 1]

    return adj


@task(returns=dict)
def counts_inlinks(adjlist1):
    """Generate a list of frequency of each vertex."""
    counts_in = {}
    for v_out in adjlist1:
        counts_in[v_out] = adjlist1[v_out][1]
    return counts_in


@task(returns=dict)
def merge_counts(counts1, counts2):
    """Merge the frequency of each vertex."""
    for v_out in counts2:
        if v_out in counts1:
            counts1[v_out] += counts2[v_out]
        else:
            counts1[v_out] = counts2[v_out]
    return counts1


@task(returns=dict)
def update_AdjList(adj1, counts_in):
    """Update the frequency of vertex in each fragment."""
    for key in adj1:
        adj1[key][1] = counts_in[key]
    return adj1


@task(returns=dict)
def calcContribuitions(adj, ranks):
    """Calculate the partial contribution of each vertex."""
    contrib = {}
    for key in adj:
        urls = adj[key][0]
        numNeighbors = adj[key][1]
        rank = ranks[key]
        for url in urls:
            if url not in contrib:
                #  destino  =  contrib
                contrib[url] = rank/numNeighbors
            else:
                contrib[url] += rank/numNeighbors

    return contrib


@task(returns=dict)
def mergeContribs(contrib1, contrib2):
    """Merge the contributions."""
    for k2 in contrib2:
        if k2 in contrib1:
            contrib1[k2] += contrib2[k2]
        else:
            contrib1[k2] = contrib2[k2]

    return contrib1


@task(returns=dict)
def updateRank_p(ranks, contrib, factor):
    """Update the rank of each vertex in the fragment."""
    bo = 1.0 - factor

    for key in contrib:
        if key in ranks:
            ranks[key] = bo + factor*contrib[key]

    return ranks


@task(returns=list)
def mergeRanks(df1, df2):
    """Merge and remove duplicates vertex."""
    result = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    result.reset_index(drop=True, inplace=True)
    return result


@task(returns=list)
def split(merged_table, numFrag):
    """Split the list of vertex into numFrag parts.

    Note: the list of vertex and their ranks must be fit in memory.
    """
    import numpy as np
    result = np.array_split(merged_table, numFrag)
    return result


@task(returns=list)
def printingResults(ranks, c1, c2):
    """Create the final result."""
    Links = []
    Ranks = []
    for v in ranks:
        Links.append(v)
        Ranks.append(ranks[v])

    data = pd.DataFrame()
    data[c1] = Links
    data[c2] = Ranks

    return data
