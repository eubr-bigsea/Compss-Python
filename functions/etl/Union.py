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
# Union of the datasets

def Union(data1, data2,numFrag):
    """
        Function which do a union between two np.arrays.
        The output remains splitted.

        :param data1: A np.array with already splited in numFrags.
        :param data2: Other np.array with already splited in numFrags.
        :return: Returns a new np.arrays.
    """

    data_result = [Union_part(data1[f], data2[f]) for f in range(numFrag)]

    return data_result


@task(returns=list)
def Union_part(list1,list2):
    print "\nUnion_part\n---\n{}\n---\n{}\n---\n".format(list1,list2)

    if len(list1) == 0:
        result = list2
    elif len(list2) == 0:
        result = list1
    else:
        result = pd.concat([list1,list2], ignore_index=True)
    return  result
#-------------------------------------------------------------------------------
