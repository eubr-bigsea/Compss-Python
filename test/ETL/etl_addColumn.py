#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.add_columns import AddColumnsOperation
import pandas as pd

# pd.set_option('display.expand_frame_repr', False)

if __name__ == '__main__':
    """Test AddColumns function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['header'] = True
    settings['separator'] = ','

    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)
    # data0 = compss_wait_on(data0)
    #
    data1 = data0[:]
    #
    # for f in range(numFrag):
    #     data1[f] = data0[f][0:100]

    suffixes = ['_l', '_r']
    data1 = AddColumnsOperation().transform(data0, data1,
                                            suffixes, numFrag)

    data1 = compss_wait_on(data1)

    data1 = pd.concat(data1, axis=0)
    print data1[['Year_l', 'Year_r']].values

