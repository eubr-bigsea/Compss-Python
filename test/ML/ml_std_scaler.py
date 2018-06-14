#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.ml.standard_scaler import StandardScalerOperation
from functions.ml.feature_assembler import FeatureAssemblerOperation
import pandas as pd

# pd.set_option('display.expand_frame_repr', False)


if __name__ == '__main__':
    """Test Stardard Scaler function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['separator'] = ','
    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)

    settings = dict()
    settings['cols'] = ['ActualElapsedTime', 'CRSElapsedTime']
    settings['alias'] = 'FEATURES'
    data0 = FeatureAssemblerOperation().transform(data0, settings, numFrag)

    settings = dict()
    settings['attributes'] = ['FEATURES', 'FEATURES']
    settings['alias'] = ['Scaled1', 'Scaled2']
    data1 = StandardScalerOperation().transform(data0, settings, numFrag)

    data1 = compss_wait_on(data1)

    data1 = pd.concat(data1, axis=0)
    print data1[['FEATURES', 'Scaled1', 'Scaled2']].head(50)
