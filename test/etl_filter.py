#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.etl.Filter import FilterOperation

if __name__ == '__main__':
    """Test filter function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)
    data0 = compss_wait_on(data0)
    count = sum([len(d) for d in data0])

    settings = dict()
    settings['query'] = "(FlightNum == 4)"
    data1 = FilterOperation().transform(data0, settings, numFrag)
    data1 = compss_wait_on(data1)
    count2 = sum([len(d) for d in data1])

    print count2 < count
