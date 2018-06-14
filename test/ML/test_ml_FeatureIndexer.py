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
from functions.ml.FeatureIndexer import FeatureIndexerOperation

pd.set_option('display.expand_frame_repr', False)


def main( ):

    # From HDFS
    numFrag = 4
    settings = dict()
    settings['port'] = 0
    settings['host'] = 'default'
    settings['path'] = '/iris_categorico.csv'
    settings['header'] = True
    settings['separator'] = ','

    data = ReadCSVFromHDFSOperation(settings, numFrag)

    # String ---> Index
    settings = dict()
    settings['inputCol'] = 'label'
    settings['outputCol'] = 'INDEXED'

    data1, model = FeatureIndexerOperation(data,settings,numFrag)

    # Index ---> String
    settings['model'] = model
    settings['IndexToString'] = True
    settings['inputCol'] = 'INDEXED'
    settings['outputCol'] = 'label_AGAIN'
    data = FeatureIndexerOperation(data1,settings,numFrag)

    data = compss_wait_on(data)
    for d in data:
        print d

main( )
