#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Auto-generated COMPSs code from Lemonade Workflow
# (c) Speed Labs - Departamento de Ciência da Computação
#     Universidade Federal de Minas Gerais
# More information about Lemonade to be provided
#
import json
import os
import re
import string
import sys
import time
import unicodedata
import csv
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.api.api import compss_wait_on



from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.data.SaveData import SaveOperation
from functions.ml.classification.Knn.knn import KNN
from functions.ml.FeatureAssembler import FeatureAssemblerOperation

reload(sys)
sys.setdefaultencoding('utf8')

pd.set_option('display.expand_frame_repr', False)


def main( ):
    """ Run generated code """
    numFrag = 4
    settings = dict()
    settings['port'] = 0
    settings['host'] = 'default'
    settings['path'] = '/iris_categorico.csv'
    settings['separator'] = ','

    data0 = ReadCSVFromHDFSOperation(settings, numFrag)

    data0 = compss_wait_on(data0)

    data0[2] = data0[2][0:0]
    # for d in data0:
    #     print d

    #Create a FeatureAssembled
    data0 = FeatureAssemblerOperation(data0, ['x','y'], 'FEATURES', numFrag)

    data0 = compss_wait_on(data0)
    for d in data0:
        print d

    settings = dict()
    settings['K'] = 2
    settings['features'] = 'FEATURES'
    settings['label'] = 'label'
    settings['predCol'] = 'PREDICTED_LABEL'
    knn = KNN()
    data1 = knn.fit_transform(data0, settings, numFrag)

    from functions.ml.metrics.ClassificationModelEvaluation import ClassificationModelEvaluation

    cls = ClassificationModelEvaluation()

    settings = dict()
    settings['pos_label'] = 'A'
    settings['test_col'] = 'label'
    settings['pred_col'] = 'PREDICTED_LABEL'
    settings['binary'] = True
    data1 = cls.calculate(data1,settings,numFrag)

    data1 = compss_wait_on(data1)
    print data1



    # numFrag = 4
    # settings = dict()
    # settings['filename'] = 'output.csv'
    # settings['mode'] = 'append'
    # settings['header'] = True
    # settings['format'] = 'CSV'
    # data1_tmp = SaveOperation(data1, settings, numFrag)

main( )
