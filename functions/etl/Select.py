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
# Select/Projection columns in a array

def SelectOperation(data,columns,numFrag):
    """
        Function which do a Projection with the columns choosed.
        The output remains splitted.

        :param data: A np.array with already splited in numFrags.
        :param columns: A list with the indexs which will be selected.
        :return: Returns a np.array with only the columns choosed.
    """

    data = [Select_part(data[f],columns) for f in range(numFrag)]

    return data

@task(returns=list)
def Select_part(list1,fields):
    return list1[fields]

#-------------------------------------------------------------------------------
