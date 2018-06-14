#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.ml.classification.LogisticRegression.logisticRegression import logisticRegression
from functions.ml.feature_assembler import FeatureAssemblerOperation
import pandas as pd

pd.set_option('display.expand_frame_repr', False)

if __name__ == '__main__':
    """Test Logistic Regression function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/iris_dataset.csv'
    settings['separator'] = ','
    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)


    settings = dict()
    settings['cols'] = ['x', 'y']
    settings['alias'] = 'FEATURES'
    data0 = FeatureAssemblerOperation().transform(data0, settings, numFrag)

    settings = dict()
    settings['iters'] = 10
    settings['threshold'] = 0.001
    settings['regularization'] = 0.1
    settings['alpha'] = 0.1
    settings['features'] = 'FEATURES'
    settings['label'] = 'label'
    settings['predCol'] = 'PREDICTED_LABEL'

    lr = logisticRegression()
    model = lr.fit(data0, settings, numFrag)
    data1 = lr.transform(data0, model, settings, numFrag)

    data1 = compss_wait_on(data1)
    data1 = pd.concat(data1, axis=0)
    print data1[['label', 'PREDICTED_LABEL']]