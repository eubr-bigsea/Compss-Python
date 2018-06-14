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



from functions.data.ReadData import ReadCSVFromHDFSOperation, ReadCSVOperation
from functions.data.SaveData import SaveOperation
from functions.ml.FeatureAssembler import FeatureAssemblerOperation
from functions.ml.regression.linearRegression.linearRegression import linearRegression
pd.set_option('display.expand_frame_repr', False)


def main( ):

    LINEAR_1D = False
    LINEAR_3D = True
    if LINEAR_1D:
        # From HDFS
        numFrag = 4
        settings = dict()
        settings['port'] = 0
        settings['host'] = 'default'
        settings['path'] = '/1D_linearRegression.txt'
        settings['header'] = True
        settings['separator'] = ','

        data = ReadCSVFromHDFSOperation(settings, numFrag)

        data = compss_wait_on(data)
        data[2] = data[2][0:0]

        lr = linearRegression()
        settings = dict()
        settings['features'] = 'x_norm'
        settings['label'] = 'y_norm'
        settings['mode'] = "simple" #'simple' #simple
        settings['predCol'] = "PREDICTED"
        settings['max_iter'] = 15
        settings['alpha'] = 0.001
        model = lr.fit(data,settings,numFrag)
        data1 = lr.transform(data,model,settings,numFrag)



    if LINEAR_3D:
        # From HDFS
        numFrag = 4
        settings = dict()
        settings['port'] = 0
        settings['host'] = 'default'
        settings['path'] = '/3D_linearRegression.txt'
        settings['header'] = True
        settings['separator'] = ','

        data = ReadCSVFromHDFSOperation(settings, numFrag)

        data = compss_wait_on(data)
        data[2] = data[2][0:0]

        data1 = FeatureAssemblerOperation(data, ['x_norm','y_norm'], 'FEATURES', numFrag)

        lr = linearRegression()
        settings = dict()
        settings['features'] = 'FEATURES'
        settings['label'] = 'z_norm'
        settings['mode'] = "SDG"
        settings['predCol'] = "PREDICTED"
        settings['max_iter'] = 3
        settings['alpha'] = 0.01
        model = lr.fit(data1,settings,numFrag)

        data1 = lr.transform(data1,model,settings,numFrag)



    from functions.ml.metrics.RegressionModelEvaluation import RegressionModelEvaluation

    rme = RegressionModelEvaluation()
    settings = {}
    settings['metric'] = 'MSE'
    settings['pred_col'] = 'PREDICTED'
    settings['test_col'] = 'z_norm'
    settings['features'] = 'FEATURES'#'x_norm'

    data = rme.calculate(data1,settings,numFrag)
    data = compss_wait_on(data)
    print data
main( )
