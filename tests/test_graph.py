#!/usr/bin/python
# -*- coding: utf-8 -*-
from ddf_library.context import COMPSsContext


def graph_pagerank():

    from ddf_library.functions.graph import PageRank
    data1 = COMPSsContext()\
        .read.csv('hdfs://localhost:9000/edgelist-pagerank.csv',
                  num_of_parts=4)\
        .select(['inlink', 'outlink'])

    result = PageRank(max_iters=7)\
        .transform(data1, inlink_col='inlink', outlink_col='outlink')\
        .select(['Vertex', 'Rank'])

    print("RESULT :\n")
    result.show()


if __name__ == '__main__':
    print("_____Testing Graph - PageRank_____")
    graph_pagerank()
