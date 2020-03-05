#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.bases.metadata import Status, OPTGroup
from ddf_library.bases.context_base import ContextBase
from ddf_library.ddf import DDF
from ddf_library.bases.ddf_model import ModelDDF
from ddf_library.utils import generate_info, read_stage_file, \
    create_stage_files, save_stage_file

from pycompss.api.api import compss_wait_on, compss_delete_object
from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.parameter import FILE_IN, COLLECTION_IN

import pandas as pd
import numpy as np

__all__ = ['PageRank']

# TODO: this algorithm can be optimized


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

        compss_delete_object(counts_in)

        for iteration in range(self.max_iters):
            """Calculate the partial contribution of each vertex."""
            contributions = [_calc_contribuitions(adj_list[i], rank_list[i])
                             for i in range(nfrag)]

            merged_c = merge_reduce(_merge_counts, contributions)

            """Update each vertex rank in the fragment."""
            rank_list = [_update_rank(rank_list[i], merged_c,
                                      self.damping_factor)
                         for i in range(nfrag)]

        merged_table = merge_ranks(rank_list, col1, col2)
        result, info = _pagerank_split(merged_table, nfrag)

        new_state_uuid = ContextBase\
            .ddf_add_task(self.name,
                          status=Status.STATUS_COMPLETED,
                          opt=OPTGroup.OPT_OTHER,
                          info_data=info,
                          parent=[tmp.last_uuid],
                          result=result,
                          function=[self.transform, data])

        return DDF(last_uuid=new_state_uuid)


@task(returns=3, data=FILE_IN)
def _pr_create_adjlist(data, inlink, outlink):
    cols = [outlink, inlink]
    adj = {}
    ranks = {}

    data = read_stage_file(data, cols=cols)
    for link in data[cols].to_numpy():
        v_out, v_in = link

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
        rank = ranks[key]
        for url in urls:
            if url not in contrib:
                #  out       =  contrib
                contrib[url] = rank/num_neighbors
            else:
                contrib[url] += rank/num_neighbors

    return contrib


@task(returns=1)
def _update_rank(ranks, contrib, factor):
    """Update the rank of each vertex in the fragment."""
    bo = 1.0 - factor

    for key in contrib:
        if key in ranks:
            ranks[key] = bo + factor*contrib[key]

    return ranks


@task(returns=1, dfs=COLLECTION_IN)
def merge_ranks(dfs, c1, c2):
    """Create the final result. Merge and remove duplicates vertex."""
    dfs = [pd.DataFrame(ranks.items(), columns=[c1, c2]) for ranks in dfs]

    dfs = pd.concat(dfs, ignore_index=True)\
        .drop_duplicates(ignore_index=True)\
        .sort_values(['Rank'], ascending=False, ignore_index=True)

    return dfs


def _pagerank_split(result, nfrag):
    """Split the list of vertex into nfrag parts.

    Note: the list of unique vertex and their ranks must be fit in memory.
    """
    result = compss_wait_on(result)
    result = np.array_split(result, nfrag)
    outfiles = create_stage_files(nfrag)
    info = [0] * nfrag

    for f, table in enumerate(result):
        save_stage_file(outfiles[f], table)
        info[f] = generate_info(table, f)

    return outfiles, info



