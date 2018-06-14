#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.ml.pca import PCA
from functions.ml.feature_assembler import FeatureAssemblerOperation
import pandas as pd


if __name__ == '__main__':
    """Test PCA function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/iris_dataset.csv'
    settings['separator'] = ','
    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)

    settings = dict()
    settings['cols'] = ['Year', 'Month', 'Day',
                        'CRSElapsedTime']
    settings['alias'] = 'FEATURES'
    data0 = FeatureAssemblerOperation().transform(data0, settings, numFrag)

    settings = dict()
    settings['features'] = 'FEATURES'
    settings['predCol'] = 'PCA'
    settings['NComponents'] = 1
    pca = PCA()
    model = pca.fit(data0, settings, numFrag)
    print "*" * 20
    print model
    print "*" * 20
    data = pca.transform(data0, model, settings, numFrag)
    data = compss_wait_on(data)
    old = pd.concat(data, axis=0)
    print old[['CRSElapsedTime', 'PCA']]
