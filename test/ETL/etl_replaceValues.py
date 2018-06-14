#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.replace_values import ReplaceValuesOperation
import pandas as pd

if __name__ == '__main__':
    """Test ReplaceValues function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadOperationHDFS.transform(filename, settings, numFrag)

    settings = dict()
    settings['replaces'] = {'Year': [[2008], [-1]],
                            'Dest': [['LAS', 'BUR'], ["L", 'B']]}
    data1 = ReplaceValuesOperation().transform(data0, settings, numFrag)
    data1 = compss_wait_on(data1)
    data1 = pd.concat(data1, axis=0)

    values = data1['Dest'].values
    print (data1.iloc[0]['Year'] == -1) and\
          (all(d not in values for d in ['LAS', 'BUR']))
