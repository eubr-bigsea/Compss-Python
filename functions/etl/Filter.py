#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"




from pycompss.api.task import task
from pycompss.api.parameter import *

import numpy as np
import pandas as pd


#-------------------------------------------------------------------------------
#   Filter

def FilterOperation(data, settings, numFrag):

    result = [[] for i in range(numFrag)]
    for i in range(numFrag):
        result[i] = filter_partial(data[i], settings)
    return result


@task(returns=list)
def filter_partial(data, settings):
    #test if row_condition is valid
    row_condition = settings['query']
    data.query(row_condition, inplace=True)
    return data
