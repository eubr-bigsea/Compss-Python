#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from pycompss.api.api import compss_wait_on


from functions.data.read_data import ReadOperationHDFS
from functions.ml.feature_assembler import FeatureAssemblerOperation

pd.set_option('display.expand_frame_repr', False)


def main( ):
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['separator'] = ','
    filename = '/iris_categorico.csv'

    data = ReadOperationHDFS().transform(filename, settings, numFrag)

    data = compss_wait_on(data)
    data[2] = data[2][0:0]
    
    data1 = FeatureAssemblerOperation().transform(data, ['x','y'], 'FEATURES', numFrag)
    data1 = FeatureAssemblerOperation().transform(data1, ['label', 'FEATURES'], 'FEATURES', numFrag)

    data1 = compss_wait_on(data1)
    for d in data1:
        print d

main( )
