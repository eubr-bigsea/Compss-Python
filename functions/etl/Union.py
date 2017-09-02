#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *

import numpy as np
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

    data_result = [Union_part(data1[f], data2[f]) for f in range(numFrag)]

    return data_result


@task(returns=list)
def Union_part(list1,list2):

    if len(list1) == 0:
        result = list2
    elif len(list2) == 0:
        result = list1
    else:
        result = pd.concat([list1,list2], ignore_index=True)
    return  result
#-------------------------------------------------------------------------------
