#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.etl.Drop import DropOperation

if __name__ == '__main__':
    """Test drop function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)
    data0 = compss_wait_on(data0)

    to_remove = ['Year', 'Month', 'DayofMonth']
    data1 = DropOperation().transform(data0, to_remove, numFrag)
    data1 = compss_wait_on(data1)
    cols = [d.columns for d in data1]
    diff_cols = any([set(cols[0]) != set(cols[i]) for i in xrange(1, numFrag)])
    removed = all(x in cols[0] for x in to_remove)
    print not removed and not diff_cols
