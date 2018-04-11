#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.etl.Split import SplitOperation

if __name__ == '__main__':
    """Test split function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)

    settings = dict()
    settings['percentage'] = 0.25
    data1, data2 = SplitOperation().transform(data0, settings, numFrag)

    data0 = compss_wait_on(data0)
    data1 = compss_wait_on(data1)
    data2 = compss_wait_on(data2)

    total = sum([len(d) for d in data0])
    count1 = sum([len(d) for d in data1]) + sum([len(d) for d in data2])
    print count1 == total
