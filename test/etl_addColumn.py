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

from timeit import default_timer as timer
from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.api.api import compss_wait_on



from functions.data.ReadData import ReadCSVFromHDFSOperation
from functions.data.SaveData import SaveOperation
from functions.etl.AddColumns import AddColumnsOperation

reload(sys)
sys.setdefaultencoding('utf8')


import pandas as pd
import numpy as np

def main( ):
    """ Run generated code """
    numFrag = 8
    settings = dict()
    settings['port'] = 0
    settings['host'] = 'default'
    settings['path'] = '/iris_categorico.csv'
    settings['header'] = True
    settings['separator'] = ','
    settings['infer'] = 'NO'
    settings['mode'] = 'FAILFAST'
    settings['na_values'] = ['']
    #data0 = ReadCSVFromHDFSOperation(settings, numFrag)

    settings['path'] = '/location_data_head100.csv'
    #data1_1 = ReadCSVFromHDFSOperation(settings, numFrag)

    off = 10
    data = [[] for f in range(numFrag)]
    start = 0
    for f in range(numFrag):
        data[f] = pd.DataFrame(np.arange(start, start+off))
        start+=off

    off = 10
    data1_1 = [[] for f in range(numFrag)]
    start = 0
    for f in range(numFrag):
        data1_1[f] = pd.DataFrame(np.arange(start, start+off))
        start+=off


    # CASE 1
    data[0] = pd.DataFrame(np.arange(0, 0))
    data[1] = pd.DataFrame(np.arange(0, 5))
    data[2] = pd.DataFrame(np.arange(5, 20))
    data[3] = pd.DataFrame(np.arange(20, 40))
    data[4] = pd.DataFrame(np.arange(40, 50))
    data[5] = pd.DataFrame(np.arange(50, 57))
    data[6] = pd.DataFrame(np.arange(57, 57))
    data[7] = pd.DataFrame(np.arange(57, 67))

    data1_1[0] = pd.DataFrame(np.arange(0, 10))
    data1_1[1] = pd.DataFrame(np.arange(10, 20))
    data1_1[2] = pd.DataFrame(np.arange(20, 30))
    data1_1[3] = pd.DataFrame(np.arange(30, 40))
    data1_1[4] = pd.DataFrame(np.arange(40, 40))
    data1_1[5] = pd.DataFrame(np.arange(40, 43))
    data1_1[6] = pd.DataFrame(np.arange(43, 53))
    data1_1[7] = pd.DataFrame(np.arange(53, 55))


    data1_1[0] = pd.DataFrame(np.arange(0, 10))
    data1_1[1] = pd.DataFrame(np.arange(10, 20))
    data1_1[2] = pd.DataFrame(np.arange(20, 30))
    data1_1[3] = pd.DataFrame(np.arange(30, 40))
    data1_1[4] = pd.DataFrame(np.arange(40, 40))
    data1_1[5] = pd.DataFrame(np.arange(40, 43))
    data1_1[6] = pd.DataFrame(np.arange(43, 53))
    data1_1[7] = pd.DataFrame(np.arange(53, 60))



    # case 1
    data[0] = pd.DataFrame(np.arange(0, 0))
    data[1] = pd.DataFrame(np.arange(0, 5))
    data[2] = pd.DataFrame(np.arange(5, 15))
    data[3] = pd.DataFrame(np.arange(15, 20))
    data[4] = pd.DataFrame(np.arange(20, 20))
    data[5] = pd.DataFrame(np.arange(20, 27))
    data[6] = pd.DataFrame(np.arange(27, 30))
    data[7] = pd.DataFrame(np.arange(30, 30))


    suffixes = ['_e','_d']
    data1 = AddColumnsOperation(data,data1_1,False,suffixes,numFrag)

    data1 = compss_wait_on(data1)


    sum1 = 0
    sum2 = 0
    sum3 = 0
    for d1,d2,d3 in zip(data,data1_1,data1):
        print '{} '.format(d3)
        print 'len({}) len({}) len({}) '.format(len(d1),len(d2),len(d3))
        sum2+= len(d2)
        sum1+= len(d1)
        sum3+= len(d3)
        print '------------'
    print sum1
    print sum2
    print sum3
    # numFrag = 4
    # settings = dict()
    # settings['filename'] = 'output.csv'
    # settings['mode'] = 'append'
    # settings['header'] = True
    # settings['format'] = 'CSV'
    # data1_tmp = SaveOperation(data1, settings, numFrag)

main( )
