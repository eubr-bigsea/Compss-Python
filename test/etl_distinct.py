#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.etl.Distinct import DistinctOperation

if __name__ == '__main__':
    """Test Distinct function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)

    columns = ['Year']
    data1 = DistinctOperation().transform(data0, columns, numFrag)
    data1 = compss_wait_on(data1)

    total = sum([len(d) for d in data1])
    print total == 1
