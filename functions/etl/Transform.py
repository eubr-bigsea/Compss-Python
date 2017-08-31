#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.api.parameter import *

import numpy as np
import pandas as pd
import datetime
import time


#-------------------------------------------------------------------------------
#  Transformation

def TransformOperation(data,settings,numFrag):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - functions:    a list with the lambda function and the alias
            - function:     the function to be applied;
            - new_column:   the aliases to each function applied.
            - import:       the import if exists

    ex.:   settings['functions'] = ['alias_col1', "lambda row: row['col1'].lower()", None]

    return:   the same dataframe with the new columns
    """

    functions =  settings.get('functions', [])
    if len(functions)>0:
        result = [apply_transformation(data[f], functions) for f in range(numFrag)]
        return result
    else:
        return data



@task(returns=list)
def apply_transformation(data, functions):

    for ncol, function in functions:
        if imp != '':
            exec(imp)
        print function
        data[ncol] = data.apply(eval(function), axis=1)
    #print data
    return data


# def group_datetime(d, interval):
#     seconds = d.second + d.hour*3600 + d.minute*60 + d.microsecond/1000
#     k = d - datetime.timedelta(seconds=seconds % interval)
#     return datetime.datetime(k.year, k.month, k.day, k.hour, k.minute, k.second)
