#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd

def DistinctOperation(data, cols, numFrag):
    """
        DistinctOperation():
        Function which remove duplicates elements (distinct elements) in a
        pandas dataframe.

        :param data:        A list with numFrag pandas's dataframe;
        :param cols:        A list with the columns names to take in count
                            (if no field is choosen, all fields are used).
        :param numFrag:     The number of fragments;
        :return:            Returns a list with numFrag pandas's dataframe.
    """

    buff = [(f,g) for f in range(numFrag)  for g in xrange(f,numFrag) if f != g]

    def disjoint(a, b):
        return  set(a).isdisjoint(b)

    while len(buff)>0:
        step_list_i = []
        step_list_j = []
        step_list_i.append(buff[0][0])
        step_list_j.append(buff[0][1])
        del buff[0]

        for i in range(len(buff)):
            tuples = buff[i]
            if  disjoint(tuples, step_list_i):
                if  disjoint(tuples, step_list_j):
                    step_list_i.append(tuples[0])
                    step_list_j.append(tuples[1])
                    del buff[i]

        for x,y in zip(step_list_i,step_list_j):
            DropDuplicates_p(data[x],data[y],cols)


    return data

@task(data1=INOUT,data2=INOUT)
def DropDuplicates_p(data1,data2,cols):
    data = pd.concat([data1,data2],axis=0, ignore_index=True)

    #if no field is choosen, all fields are used)
    if len(cols)==0:
        cols = data.columns

    data.drop_duplicates(cols, keep='first',inplace=True)
    data = np.array_split(data, 2)

    data[0].reset_index(drop=True,inplace=True)
    data[1].reset_index(drop=True,inplace=True)
    data1.ix[0:] =  data[0].ix[0:]
    data2.ix[0:] =  data[1].ix[0:]

    data1.dropna(axis=0,inplace=True,subset=None,how='all')
    data2.dropna(axis=0,inplace=True,subset=None,how='all')
