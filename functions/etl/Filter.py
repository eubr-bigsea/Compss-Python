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
#   Filter



def makecondition(settings,cols):
    #struct_cond = np.array([["a","=",0],["a","=",0],["a","=",0]])
    struct_cond = np.array(settings['row_condition'])
    all_condition = ["=","!=","<",">",">=","<="]
    if not set(struct_cond[:,0]).issubset(set(cols)): #at least one colum doesnt exists
        return ""
    elif not set(struct_cond[:,1]).issubset(set(all_condition)):
        return ""
    else:
        query = ""
        for c in struct_cond:
            query += "({} {} {}) &".format(c[0],c[1],c[2])
        query = query[:-1]
        return query


def Filter(data,settings,numFrag):
    partial_m = [[] for i in range(numFrag)]
    for i in range(numFrag):
        partial_m[i] = filter_partial(data[i],settings)
    return partial_m


@task(returns=list)
def filter_partial(data,row_condition):
    # [[col,equality,value]]
    cols = data.columns
    row_condition = makecondition(settings,cols)

    return data.query(row_condition)
