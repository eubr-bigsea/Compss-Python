#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.graph.PageRank.pagerank import PageRank

import pandas as pd
pd.set_option('display.expand_frame_repr', False)


def main():

    # From HDFS
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/edgelist_PageRank.csv'
    settings['header'] = True
    settings['separator'] = ','

    data = ReadOperationHDFS().transform(filename, settings, numFrag)

    settings = dict()
    settings['inlink'] = 'inlink'
    settings['outlink'] = 'outlink'
    settings['maxIters'] = 10

    pr = PageRank()
    data = pr.transform(data, settings, numFrag)

    data = compss_wait_on(data)
    data = pd.concat(data, axis=0, sort=False)
    print data


main()


