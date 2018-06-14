#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
import pandas as pd

if __name__ == '__main__':
    """Test data reader function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['separator'] = ','
    # settings['format'] = 'json'
    # filename = '/sample_doc1.json'
    filename = '/flights.csv'

    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)

    data0 = compss_wait_on(data0)
    data0 = pd.concat(data0, sort=False)
    count = len(data0)

    print data0

