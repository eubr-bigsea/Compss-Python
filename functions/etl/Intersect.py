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
# Intersect

def IntersectionOperation(data1,data2,numFrag):
    """
        Function which returns a new set with elements that are common to all sets.
        The output is already merged.

        :param data1: A np.array with already splited in numFrags
        :param data2: A np.array with already splited in numFrags
        :return: Returns a  new np.array
    """

    data_result = [[] for i in range(numFrag)]

    for i in xrange(numFrag):
        data_partial   = [ Intersect_part(data1[i], data2[j]) for j in xrange(numFrag) ]
        data_result[i] =  mergeReduce(Union_part,data_partial)

    return data_result


@task(returns=list)
def Intersect_part(list1,list2):
    print "\nIntersect_part\n---\n{}\n---\n{}\n---\n".format(list1,list2)
    if len(list1) == 0 or len(list2) == 0:
        result = []
    else:
        result = list1.merge(list2)
        print result

    return  result
#-------------------------------------------------------------------------------
