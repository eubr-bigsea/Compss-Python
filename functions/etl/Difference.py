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
#   Difference

def DifferenceOperation(data1,data2,numFrag):
    """
        Function which returns a new set with containing rows in the first frame
        but not in the second one.
        The output is already merged.

        :param data1: A np.array with already splited in numFrags
        :param data2: A np.array with already splited in numFrags
        :return: Returns a  new np.array
    """
    from pycompss.api.api import compss_wait_on

    data_result = [pd.DataFrame() for i in range(len(data1))]

    for f1 in range(len(data1)):
        data_partial      = [ Difference_part(data1[f1], data2[f2]) for f2 in range(numFrag) ]
        data_result[f1]  = mergeReduce(Intersect_part, data_partial)

    #data_result  = mergeReduce(Union_part,data_partial)
    #data_result  = compss_wait_on(data_result)

    return data_result

@task(returns=list)
def Difference_part(df1,df2):
    # a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    # a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    # result = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    if len(df1) > 0:
        if len(df2) > 0:
            names = df1.columns
            ds1 = set([ tuple(line) for line in df1.values.tolist()])
            ds2 = set([ tuple(line) for line in df2.values.tolist()])
            result = pd.DataFrame(list(ds1.difference(ds2)))
            result.columns = names
            print result
            return result
        else:
            return df2
    else:
        return pd.DataFrame()
#-------------------------------------------------------------------------------
