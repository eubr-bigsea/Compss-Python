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
from functions.ml.models import SaveModelToHDFS,LoadModelFromHDFS
from functions.ml.FeatureAssembler import FeatureAssemblerOperation


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

    #creating a fake model
    Model_to_save = dict()
    Model_to_save['features'] = 'FEATURES'
    Model_to_save['label']    = 'label'
    Model_to_save['predCol'] = 'PREDICTED_LABEL'
    Model_to_save['coef_maxIters'] = 5
    Model_to_save['model'] = data0[0]



    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['path'] = '/modelo.pkl'
    settings['overwrite'] = True

    sucess = SaveModelToHDFS(Model_to_save, settings)

    sucess = compss_wait_on(sucess)
    print sucess

    model = LoadModelFromHDFS(settings)
    model = compss_wait_on (model)
    print "-" * 40
    print model
    print "-" * 40

main( )
