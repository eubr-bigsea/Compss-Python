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
    """
    FilterOperation():

    :param data:       A list with numFrag pandas's dataframe;
    :param settings:   A dictionary that contains:
        - 'query':     A valid query.
    :param numFrag:    The number of fragments;
    :return:           Returns a list with numFrag pandas's dataframe.


    Note: Visit the link bellow to more information about the query.
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html

    example:
        settings['query'] = "(VEIC == 'CARX')" to rows where VEIC is CARX
        settings['query'] = "(VEIC == VEIC) and (YEAR > 2000)" to rows where
            VEIC is not NaN and YEAR is greater than 2000
    """

    result = [[] for i in range(numFrag)]
    for i in range(numFrag):
        result[i] = filter_partial(data[i], settings)
    return result


@task(returns=list)
def filter_partial(data, settings):
    row_condition = settings.get('query',"")
    data.query(row_condition, inplace=True)
    return data
