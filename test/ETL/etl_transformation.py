#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.transform import TransformOperation

if __name__ == '__main__':
    """Test transformation function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadOperationHDFS.transform(filename, settings, numFrag)

    settings = dict()
    settings['functions'] = \
            [['SUM',
              "lambda col: col['Year']+col['Month']+col['DayofMonth']",
              '']]

    data1 = TransformOperation().transform(data0, settings, numFrag)
    data1 = compss_wait_on(data1)

    print data1[0].iloc[0]['SUM'] == 2012
