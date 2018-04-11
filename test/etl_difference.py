#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.etl.Difference import DifferenceOperation
import pandas as pd

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
    data1 = [[] for i in range(numFrag)]

    for i in range(numFrag):
        if i != 3:
            data1[i] = data0[i][0:100]
        else:
            data1[i] = data0[i][0:0]
        data1[i] = data1[i].sort_values('CRSDepTime', na_position='first')
        data1[i].reset_index(drop=True, inplace=True)

    data0[1] = data0[1][0:0]

    data2 = DifferenceOperation().transform(data0, data1, numFrag)
    data2 = compss_wait_on(data2)

    count0 = sum([len(d) for d in data0])
    count1 = sum([len(d) for d in data1])
    count2 = sum([len(d) for d in data2])

    print (count0 - count1 + 100) == count2
