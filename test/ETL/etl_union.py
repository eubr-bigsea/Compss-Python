#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.etl.union import UnionOperation

if __name__ == '__main__':
    """Test Union function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)
    data0 = compss_wait_on(data0)

    data1 = [[] for i in range(numFrag)]
    count1 = 0
    total = 0
    for i in range(numFrag):
        if i == 0:
            data1[i] = data0[i][0:0]
        else:
            data1[i] = data0[i][0:25]

        total += len(data0[i])
        count1 += len(data1[i])

    data1 = UnionOperation().transform(data0, data1, numFrag)
    data1 = compss_wait_on(data1)
    count2 = sum([len(d) for d in data1])

    print (total + count1) == count2
