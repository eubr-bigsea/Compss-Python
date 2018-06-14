#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.join import JoinOperation
import pandas as pd

# pd.set_option('display.expand_frame_repr', False)

if __name__ == '__main__':
    """Test Join function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['separator'] = ','
    filename = '/flights.csv'

    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)
    data0 = compss_wait_on(data0)

    data1 = data0[:]
    for i in xrange(0, numFrag):
        data1[i] = data0[i][0:10]
    data0[0] = data0[0][0:0]

    settings = dict()

    settings['key1'] = ["FlightNum", 'CRSArrTime']
    settings['key2'] = ["FlightNum", 'CRSArrTime']
    settings['option'] = 'inner'
    settings['keep_keys'] = True
    settings['case'] = True
    settings['sort'] = True
    data1 = JoinOperation().transform(data0, data1, settings, numFrag)
    data1 = compss_wait_on(data1)
    df = pd.concat(data1, ignore_index=True)

    print len(df) == 30 and len(df.columns) == 58

    print df


