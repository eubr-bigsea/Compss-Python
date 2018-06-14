#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.ml.clustering.Kmeans.Kmeans import Kmeans
from functions.ml.feature_assembler import FeatureAssemblerOperation
from functions.etl.select import SelectOperation
import pandas as pd
import time

pd.set_option('display.expand_frame_repr', False)

if __name__ == '__main__':
    """Test Kmeans function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['header'] = False
    settings['separator'] = ','
    filename = '/train_0.0012m_cleaned.csv'
    data0 = ReadOperationHDFS().transform(filename,settings, numFrag)

    settings = dict()
    settings['cols'] = ['col_{}'.format(i) for i in xrange(1, 29)]
    settings['alias'] = 'FEATURES'
    data0 = FeatureAssemblerOperation().transform(data0, settings, numFrag)

    columns = ['col_0', 'FEATURES']
    data0 = SelectOperation().transform(data0, columns, numFrag)

    settings = dict()
    settings['features'] = 'FEATURES'
    settings['label'] = 'col_0'
    settings['predCol'] = 'PREDICTED_LABEL'
    settings['maxIterations'] = 10
    #settings['initMode'] = 'random'
    settings['k'] = 2

    km = Kmeans()
    start = time.time()
    model = km.fit(data0, settings, numFrag)
    end = time.time()
    print 'time: {}'.format(end - start)
    data1 = km.transform(data0, model, settings, numFrag)

    data1 = compss_wait_on(data1)
    data1 = pd.concat(data1, axis=0)
    print data1[['PREDICTED_LABEL','col_0']]
