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
#  Clean Missing

def CleanMissingOperation(data,params,numFrag):
    """
    Clean missing fields from data set
    Parameters:
        - attributes: list of attributes to evaluate
        - cleaning_mode: what to do with missing values. Possible values include
          * "VALUE": replace by parameter "value"
          * "REMOVE_ROW": remove entire row

          * ! NOT IMPLEMENTED YET!  "MEDIAN": replace by median value
          * ! NOT IMPLEMENTED YET!  "MODE": replace by mode value
          * "MEAN": replace by mean value

          * "REMOVE_COLUMN": remove entire column

        - value: optional, used to replace missing values
    """

    cleaning_mode = params['cleaning_mode']

    if cleaning_mode in ['VALUE','REMOVE_ROW']:
        data = [ CleanMissing_partial(data[f],params) for f in range(numFrag)]
    else:
        params = [ analysing_missingvalues(data[f],params) for f in range(numFrag)]
        params = mergeReduce(mergeCleanOptions,params)
        data = [ CleanMissing_partial(data[f],params) for f in range(numFrag)]

    return data

@task(returns=dict)
def analysing_missingvalues (data, params):
    attributes    = params['attributes']
    cleaning_mode = params['cleaning_mode']
    if cleaning_mode == "REMOVE_COLUMN":
        null_fields = data.columns[data[attributes].isnull().any()].tolist()
        params['columns_drop'] = null_fields

    elif cleaning_mode == "MEAN":

        params['values'] = data[attributes].mean().values
        print params['values']

    return params

@task(returns=dict)
def mergeCleanOptions(params1,params2):
    cleaning_mode = params1['cleaning_mode']
    if cleaning_mode == "REMOVE_COLUMN":
        params1['columns_drop'] = list(set(params1['columns_drop'] + params2['columns_drop']))
    elif cleaning_mode in "MEAN":
        params1['values'] = [ (x + y)/2 for x, y in zip(params1['values'], params2['values']) ]



    return params1

@task(returns=list)
def CleanMissing_partial(data,params):
    attributes    = params['attributes']
    cleaning_mode = params['cleaning_mode']

    if cleaning_mode == "REMOVE_ROW":   #ok
        data.dropna(axis=0, how='any', subset=attributes, inplace=True)

    elif cleaning_mode == "VALUE":  #ok
        value = params['value']
        data[attributes] = data[attributes].fillna(value=value)

    elif cleaning_mode == "REMOVE_COLUMN": #ok
        subset = params['columns_drop']
        data = data.drop(subset, axis=1)

    elif cleaning_mode in ["MEAN","MODE","MEDIAN"]:
        values = params['values']
        for v,a in zip(values,attributes):
            data[a] = data[a].fillna(value=v)


    data.reset_index(drop=True, inplace=True)
    return data
