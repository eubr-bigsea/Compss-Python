#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.etl.Aggregation import AggregationOperation

if __name__ == '__main__':
    """Test Aggregation function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)
    settings = {}
    settings['columns'] = ["Year"]
    settings['operation'] = {"Year": [u'count']}
    settings['aliases'] = {"Year": [u'COUNT']}

    data1 = AggregationOperation().transform(data0, settings, numFrag)
    data1 = compss_wait_on(data1)
    print data1[0].iloc[0]['COUNT'] == 999
