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
		 TransformOperation():

		 Returns a new DataFrame applying the expression to the specified column.
		 :param data:      A list with numFrag pandas's dataframe;
		 :settings:        A dictionary that contains:
		    - functions:   A list with an array with 3-dimensions.
		      * 1ª position:  The lambda function to be applied as a string;
		      * 2ª position:  The alias to new column to be applied the function;
		      * 3ª position:  The string to import some needed module (if needed);
		 :return:   Returns a list with numFrag pandas's dataframe with the news columns.

		ex.:   settings['functions'] = [['alias_col1', "lambda row: row['col1'].lower()", None]]
    """

    functions =  settings.get('functions', [])
    if len(functions)>0:
        result = [apply_transformation(data[f], functions) for f in range(numFrag)]
        return result
    else:
        return data



@task(returns=list)
def apply_transformation(data, functions):

    for r in functions:
        print r
        ncol, function, imp = r
        if imp != None:
            exec(imp)
        print function
        data[ncol] = data.apply(eval(function), axis=1)
    #print data
    return data


# def group_datetime(d, interval):
#     seconds = d.second + d.hour*3600 + d.minute*60 + d.microsecond/1000
#     k = d - datetime.timedelta(seconds=seconds % interval)
#     return datetime.datetime(k.year, k.month, k.day, k.hour, k.minute, k.second)
