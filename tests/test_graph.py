#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF


def graph_pagerank():

    from ddf.functions.graph import PageRank
    data1 = DDF().load_text('/edgelist_PageRank.csv', num_of_parts=4)\
        .select(['inlink', 'outlink'])

    result = PageRank(inlink_col='inlink', outlink_col='outlink', max_iters=7)\
        .transform(data1).select(['Vertex', 'Rank'])

    print "RESULT :\n", result.cache().show()


if __name__ == '__main__':
    print "_____Testing Graph - PageRank_____"
    graph_pagerank()
