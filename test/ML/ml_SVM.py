#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.ml.classification.Svm.svm import SVM
from functions.ml.feature_assembler import FeatureAssemblerOperation
import pandas as pd

pd.set_option('display.expand_frame_repr', False)

if __name__ == '__main__':
    """Test SVM function."""
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

    settings = dict()
    settings['features'] = 'FEATURES'
    settings['label'] = 'label'
    settings['predCol'] = 'PREDICTED_LABEL'
    settings['coef_maxIters'] = 10
    svm = SVM()
    import time

    i = time.time()
    model = svm.fit(data0, settings, numFrag)
    e = time.time()

    data1 = svm.transform(data0, model, settings, numFrag)

    data1 = compss_wait_on(data1)
    data1 = pd.concat(data1, axis=0)
    print data1[['label', 'PREDICTED_LABEL']]
    print e-i
