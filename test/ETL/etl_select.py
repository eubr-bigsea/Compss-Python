#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.select import SelectOperation

if __name__ == '__main__':
    """Test Select function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadOperationHDFS.transform(filename, settings, numFrag)

    columns = ['Year']
    data1 = SelectOperation().transform(data0, columns, numFrag)

    data1 = compss_wait_on(data1)
    cols = [d.columns for d in data1]

    condition = all(['Year' in c and len(c) == 1 for c in cols])
    print condition
