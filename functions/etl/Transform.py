#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks
from pycompss.api.api import compss_wait_on, barrier

import numpy as np
import pandas as pd
import math


#-------------------------------------------------------------------------------
#  Transformation

def Transform(data,settings,numFrag):
    for f in range(numFrag):
        data[f] = transform_p(data[f], settings)
    return data

@task(returns=list)
def transform_p(data, settings):
    col = settings['column']
    ncol = settings['new_column']
    tmp = []
    for row in data[col].tolist():
        lst = row.split(",")
        tmp.append(filter(None, lst))                         # Remove trailing comma
    data[ncol] = tmp

    #print data
    return data
