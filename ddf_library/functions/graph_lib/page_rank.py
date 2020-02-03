#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.ddf import DDF
from ddf_library.bases.ddf_model import ModelDDF

from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce

import pandas as pd
import numpy as np

__all__ = ['PageRank']


# TODO: Adaptar para parquet

class PageRank(ModelDDF):
    # noinspection PyUnresolvedReferences
    """
    PageRank is one of the methods Google uses to determine a page's
    relevance or importance. The idea that Page Rank brought up was that, the
    importance of any web page can be judged by looking at the pages that link
    to it.

    PageRank can be utilized in others domains. For example, may also be used
    as a methodology to measure the apparent impact of a community.

    .. note: This parallel implementation assumes that the list of unique
        vertex can be fit in memory.

    :Example:

    >>> pr = PageRank(damping_factor=0.85)
    >>> ddf2 = pr.transform(ddf1, inlink_col='col1', outlink_col='col2')
    """

    def __init__(self, damping_factor=0.85, max_iters=100):
        """
        :param damping_factor: Default damping factor is 0.85;
        :param max_iters: Maximum number of iterations (default is 100).
        """
        super(PageRank, self).__init__()

        self.inlink_col = None
        self.outlink_col = None
        self.max_iters = max_iters
        self.damping_factor = damping_factor

    def transform(self, data, outlink_col, inlink_col):
        """
        Generates the PageRank's result.

        :param data: DDF
        :param outlink_col: Out-link vertex;
        :param inlink_col: In-link vertex;
        :return: DDF with Vertex and Rank columns
        """

        df, nfrag, tmp = self._ddf_initial_setup(data)

        self.inlink_col = inlink_col
        self.outlink_col = outlink_col

        col1 = 'Vertex'
        col2 = 'Rank'

        """
        Load all URL's from the data and initialize their neighbors.
        Initialize each page’s rank to 1.0.
        """
        adj_list = [{} for _ in range(nfrag)]
        rank_list = [{} for _ in range(nfrag)]
        counts_in = [{} for _ in range(nfrag)]

        for i in range(nfrag):
            adj_list[i], rank_list[i], counts_in[i] = \
                _pr_create_adjlist(df[i], inlink_col, outlink_col)

        counts_in = merge_reduce(_merge_counts, counts_in)
        for i in range(nfrag):
            adj_list[i] = _pr_update_adjlist(adj_list[i], counts_in)

        del counts_in

        for iteration in range(self.max_iters):
            """Calculate the partial contribution of each vertex."""
            contributions = [_calc_contribuitions(adj_list[i], rank_list[i])
                             for i in range(nfrag)]

            merged_c = merge_reduce(_merge_contribs, contributions)

            """Update each vertex rank in the fragment."""
            rank_list = [_update_rank(rank_list[i], merged_c,
                                      self.damping_factor)
                         for i in range(nfrag)]

        table = [_print_result(rank_list[i], col1, col2) for i in range(nfrag)]

        merged_table = merge_reduce(_merge_ranks, table)
        result, info = _pagerank_split(merged_table, nfrag)

        uuid_key = self._ddf_add_task(task_name=self.name,
                                      status='MATERIALIZED',
                                      opt=self.OPT_OTHER,
                                      function=[self.transform, data],
                                      result=result,
                                      parent=[tmp.last_uuid],
                                      n_output=1, n_input=1, info=info)

        return DDF(task_list=tmp.task_list, last_uuid=uuid_key)


@task(returns=3)
def _pr_create_adjlist(data, inlink, outlink):

    adj = {}
    ranks = {}
    cols = [outlink, inlink]
    for link in data[cols].values:
        v_out = link[0]
        v_in = link[1]

        # Generate a partial adjacency list.
        if v_out in adj:
            adj[v_out][0].append(v_in)
            adj[v_out][1] += 1
        else:
            adj[v_out] = [[v_in], 1]

        # Generate a partial rank list of each vertex.
        if v_out not in ranks:
            ranks[v_out] = 1.0  # Rank, contributions, main
        if v_in not in ranks:
            ranks[v_in] = 1.0

    # Generate a partial list of frequency of each vertex.
    counts_in = {}
    for v_out in adj:
        counts_in[v_out] = adj[v_out][1]

    return adj, ranks, counts_in


@task(returns=1)
def _merge_counts(counts1, counts2):
    """
    Merge the frequency of each vertex.

    .. note:: It assumes that the frequency list can be fitted in memory.
    """

    for v_out in counts2:
        if v_out in counts1:
            counts1[v_out] += counts2[v_out]
        else:
            counts1[v_out] = counts2[v_out]
    return counts1


@task(returns=1)
def _pr_update_adjlist(adj1, counts_in):
    """Update the frequency of vertex in each fragment."""
    for key in adj1:
        adj1[key][1] = counts_in[key]
    return adj1


@task(returns=1)
def _calc_contribuitions(adj, ranks):
    """Calculate the partial contribution of each vertex."""
    contrib = {}
    for key in adj:
        urls = adj[key][0]
        num_neighbors = adj[key][1]
        rank = float(ranks[key])
        for url in urls:
            if url not in contrib:
                #  out       =  contrib
                contrib[url] = rank/num_neighbors
            else:
                contrib[url] += rank/num_neighbors

    return contrib


@task(returns=1)
def _merge_contribs(contrib1, contrib2):
    """Merge the contributions."""
    for k2 in contrib2:
        if k2 in contrib1:
            contrib1[k2] += contrib2[k2]
        else:
            contrib1[k2] = contrib2[k2]

    return contrib1


@task(returns=1)
def _update_rank(ranks, contrib, factor):
    """Update the rank of each vertex in the fragment."""
    bo = 1.0 - factor

    for key in contrib:
        if key in ranks:
            ranks[key] = bo + factor*contrib[key]

    return ranks


@task(returns=1)
def _print_result(ranks, c1, c2):
    """Create the final result."""
    return pd.DataFrame(ranks.items(),  columns=[c1, c2])


@task(returns=1)
def _merge_ranks(df1, df2):
    """Merge and remove duplicates vertex."""
    result = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    result.reset_index(drop=True, inplace=True)
    return result


def _pagerank_split(merged_table, nfrag):
    """Split the list of vertex into nfrag parts.

    Note: the list of unique vertex and their ranks must be fit in memory.
    """
    merged_table = compss_wait_on(merged_table)
    info = [[merged_table.columns.tolist(), merged_table.dtypes.values, []]
            for _ in range(nfrag)]
    result = np.array_split(merged_table, nfrag)
    for f, table in enumerate(merged_table):
        info[f][2].append(len(table))

    return result, info



