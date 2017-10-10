#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *

import numpy as np
import pandas as pd

#basic imports:
import datetime
import time
from dateutil.parser import parse

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
          * 3ª position:  The string to import some needed module
                          ('' if isnt needed);
        :return:   Returns a list with numFrag pandas's dataframe with
                   the news columns.

        ex.:
        settings['functions'] = \
            [ ['alias_col1', "lambda col: np.add(col['col1'],col['col2'])", ''] ]
    """

    functions =  Validate(settings)

    result = [apply_transformation(data[f], functions) for f in range(numFrag)]
    return result


def Validate(settings):
    functions = settings.get('functions', [])
    if any([
            len(functions) == 0,
            any([True if (len(func) != 3) else False for func in functions ])
            ]):
        raise Exception('You must inform a valid `functions` parameter.')
    return functions


@task(returns=list)
def apply_transformation(data, functions):

    for function in functions:
        ncol, function, imp = function
        exec(imp)
        data[ncol] = data.apply(eval(function), axis=1)
    return data
