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

          * "MEDIAN": replace by median value
          * "MODE": replace by mode value
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

    elif cleaning_mode in ["MODE",'MEDIAN']:
        dict_mode = {}
        for att in attributes:
            dict_mode[att] = data[att].value_counts()
        params['dict_mode'] = dict_mode



    return params

@task(returns=dict)
def mergeCleanOptions(params1,params2):
    cleaning_mode = params1['cleaning_mode']

    if cleaning_mode == "REMOVE_COLUMN":
        params1['columns_drop'] = list(set(params1['columns_drop'] + params2['columns_drop']))

    elif cleaning_mode in "MEAN":
        params1['values'] = [ (x + y)/2 for x, y in zip(params1['values'], params2['values']) ]

    elif cleaning_mode in ["MODE",'MEDIAN']:
        dict_mode1 = params1['dict_mode']
        dict_mode2 = params2['dict_mode']
        dict_mode  = {}
        for att in dict_mode1:
            dict_mode[att] = pd.concat([dict_mode1[att], dict_mode2[att]], axis=1).fillna(0).sum(axis=1)
        params1['dict_mode']  = dict_mode

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

    elif cleaning_mode == "MEAN": #ok
        values = params['values']
        for v,a in zip(values,attributes):
            data[a] = data[a].fillna(value=v)

    elif cleaning_mode == "MODE":  #ok
        dict_mode = params['dict_mode']

        for att in dict_mode:
            t = dict_mode[att].max()
            mode = dict_mode[att].idxmax()
            mode = dict_mode[att].index[mode]
            data[att] = data[att].fillna(value=mode)

    elif cleaning_mode == "MEDIAN":
        dict_mode = params['dict_mode']
        for att in dict_mode:
            #print dict_mode
            total = dict_mode[att].sum()
            #print total
            if total % 2 == 0:
                m1 = total/2
                m2 = total/2 +1
                #print "m1",m1
                #print "m2",m2
                count = 0
                for i, p in enumerate(dict_mode[att]):
                    count+=p
                    if count>=m1:
                        v1 =  dict_mode[att].index[i]
                    if count>=m2:
                        v2 =  dict_mode[att].index[i]
                        break
                m = (float(v1)+float(v2))/2
            else:
                m = math.floor(float(total)/2) + 1
                count = 0
                for i, p in enumerate(dict_mode[att]):
                    count+=p
                    if count>=m:
                        v1 =  dict_mode[att].index[i]
                        break
                m = float(v1)

            data[att] = data[att].fillna(value=m)


    data.reset_index(drop=True, inplace=True)
    return data
