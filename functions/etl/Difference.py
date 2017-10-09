#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd




#-------------------------------------------------------------------------------
#   Difference

def DifferenceOperation(data1,data2,numFrag):
    """
        DifferenceOperation()
        Function which returns a new set with containing rows
        in the first frame but not in the second one.

        :param data1: A list with numFrag pandas's dataframe;
        :param data2: The second list with numFrag pandas's dataframe.
        :return:      A list with numFrag pandas's dataframe.
    """

    if all([len(data1) != numFrag, len(data2) != numFrag ]):
        raise Exception("data1 and data2 must have len equal to numFrag.")


    from pycompss.api.api import compss_wait_on

    data_result = [pd.DataFrame() for i in range(numFrag)]

    for f1 in range(len(data1)):
        data_partial = \
            [ Difference_part(data1[f1], data2[f2]) for f2 in range(numFrag) ]
        data_result[f1] = mergeReduce(Intersect_part, data_partial)

    return data_result



@task(returns=list)
def Difference_part(df1,df2):

    if len(df1) > 0:
        if len(df2) > 0:
            names = df1.columns
            df = pd.merge(df1, df2, indicator=True, how='outer',on=None)
            df = df.loc[df['_merge'] == 'left_only', names]

            return df

    return df1


@task(returns=list)
def Intersect_part(list1,list2):
    keys = list1.columns.tolist()
    result = pd.merge(list1, list2, how='inner', on=keys)

    return  result
#-------------------------------------------------------------------------------
