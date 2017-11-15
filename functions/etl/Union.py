#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *

import pandas as pd


def UnionOperation(data1, data2, numFrag):
    """
        UnionOperation():
        Function which do a union between two pandas dataframes.

        :param data1:   A list with numFrag pandas's dataframe;
        :param data2:   Other list with numFrag pandas's dataframe;
        :param numFrag: The number of fragments;
        :return:        Returns a list with numFrag pandas's dataframe.
    """
    result = [[] for f in range(numFrag)]
    for f in range(numFrag):
        result[f] = Union_part(data1[f], data2[f])

    return result


@task(returns=list)
def Union_part(list1,list2):

    if len(list1) == 0:
        return list2
    elif len(list2) == 0:
        return  list1
    else:
        return pd.concat([list1,list2], ignore_index=True)

#-------------------------------------------------------------------------------
