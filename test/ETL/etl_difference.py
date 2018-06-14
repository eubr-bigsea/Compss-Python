#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.difference import DifferenceOperation

if __name__ == '__main__':
    """Test Difference function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)
    data0 = compss_wait_on(data0)
    data1 = data0[:]

    for i in range(numFrag):
        if i != 3:
            data1[i] = data1[i][0:100]
        else:
            data1[i] = data1[i][0:0]
        data1[i] = data1[i].sort_values('CRSDepTime', na_position='first')
        data1[i].reset_index(drop=True, inplace=True)

    data0[0] = data0[0][0:0]

    data2 = DifferenceOperation().transform(data0, data1, numFrag)
    data2 = compss_wait_on(data2)
    data1 = compss_wait_on(data1)

    count0 = sum([len(d) for d in data0])
    count1 = sum([len(d) for d in data1])
    count2 = sum([len(d) for d in data2])

    print (count0 - count1 + 100) == count2
