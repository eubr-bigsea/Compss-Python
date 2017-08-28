#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce


import numpy as np
import pandas as pd
import math


#-------------------------------------------------------------------------------
# Intersect

def IntersectionOperation(data1,data2,numFrag):
    """
        Function which returns a new set with elements
        that are common to all sets.

        :param data1:  A pandas dataframe already splited in numFrags
        :param data2:  A pandas dataframe already splited in numFrags
        :return: Returns a new pandas dataframe
    """

    data_result = [[] for i in range(numFrag)]

    for i in xrange(numFrag):
        data_partial   = [ Intersect_part(data1[i], data2[j]) for j in xrange(numFrag) ]
        data_result[i] =  mergeReduce(mergeIntersect,data_partial)

    return data_result


@task(returns=list)
def Intersect_part(list1,list2):
    keys = list1.columns.tolist()
    result = pd.merge(list1, list2, how='inner', on=keys)

    return  result


@task(returns=list)
def mergeIntersect(list1,list2):
    #print "\nUnion_part\n---\n{}\n---\n{}\n---\n".format(list1,list2)
    result = pd.concat([list1,list2], ignore_index=True)
    return  result
#-------------------------------------------------------------------------------
