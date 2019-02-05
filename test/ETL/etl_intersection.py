#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.intersect import IntersectionOperation

if __name__ == '__main__':
    """Test Intersection function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['separator'] = ','
    filename = '/flights.csv'

    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)
    data0 = compss_wait_on(data0)

    data1 = [[] for i in range(numFrag)]
    for i in range(numFrag):
        data0[i] = data0[i][['Year', 'DepTime', 'CRSDepTime', 'ArrTime']]
        data1[i] = data0[i][0:50]

    data1 = IntersectionOperation().transform(data0, data1, numFrag)
    data1 = compss_wait_on(data1)

    total = sum([len(d) for d in data1])
    print total == 200