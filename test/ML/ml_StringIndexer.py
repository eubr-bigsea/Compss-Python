#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.ml.string_indexer import StringIndexerOperation
import pandas as pd

pd.set_option('display.expand_frame_repr', False)


if __name__ == '__main__':
    """Test String Indexer function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['separator'] = ','
    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)
    
    # String ---> Index
    settings = dict()
    settings['inputCol'] = 'TailNum'
    settings['outputCol'] = 'INDEXED'

    data1, model = StringIndexerOperation().transform(data0, settings,numFrag)

    data1 = compss_wait_on(data1)
    print data1
    data1 = pd.concat(data1, axis=0)
    print data1[['TailNum', 'INDEXED']].values.tolist()
