#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *

import numpy as np
import pandas as pd



def SelectOperation(data,columns,numFrag):
    """
        SelectOperation():
        Function which do a Projection with the columns choosed.

        :param data:    A list with numFrag pandas's dataframe;
        :param columns: A list with the columns names which will be selected;
        :param numFrag: A number of fragments;
        :return:        A list with numFrag pandas's dataframe
                        with only the columns choosed.
    """

    data = [Select_part(data[f],columns) for f in range(numFrag)]

    return data

@task(returns=list)
def Select_part(list1,fields):
    return list1[fields]

#-------------------------------------------------------------------------------
