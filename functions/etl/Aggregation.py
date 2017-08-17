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
#  Aggregation

def AggregationOperation(data,params,numFrag):
    """
        Add one of more lines with attribute to be used, function and alias
        to compute aggregate function over groups.

        Average (AVG)	Computes the average of each group
        Count	Counts the total of records of each group
        First	Returns the first element of group
        Last	Returns the last element of group
        Maximum (MAX)	Returns the max value of each group for one attribute
        Minimum (MIN)	Returns the min value of each group for one attribute
        Sum	Returns the sum of values of each group for one attribute
    """
    columns = params['columns']
    target = params['aliases']
    operation = params['operation']


    data = [ aggregate_partial(data[f], columns, operation, target) for f in range(numFrag)]

    buff = [(f,g) for f in range(numFrag)  for g in xrange(f,numFrag) if f != g]

    new_operations = {}
    for k in target:
        values = target[k] # list of aliases
        for i in range(len(values)):
            new_operations[values[i]] = operation[k][i]


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
            merge_aggregate(data[x], data[y], columns, new_operations, target)



    return data

@task(returns=list)
def aggregate_partial(data, columns, operation, target):
    data = data.groupby(columns).agg(operation)

    newidx = []
    for (n1,n2) in data.columns.ravel():
        if len(n2)>0:
            i = operation[n1].index(n2)
            new_name = target[n1][i]
            newidx.append(new_name)
        else:
            newidx.append(n1)

    data.columns = newidx
    data = data.reset_index()


    return data


@task(data1=INOUT, data2=INOUT)
def merge_aggregate(data1, data2, columns, operation, target):
    data = pd.concat([data1,data2],axis=0, ignore_index=True)
    data = data.groupby(columns).agg(operation)

    data = data.reset_index()

    data = np.array_split(data, 2)

    data[0].reset_index(drop=True,inplace=True)
    data[1].reset_index(drop=True,inplace=True)

    data1.columns = data[0].columns
    data2.columns = data[1].columns
    data1.ix[0:] =  data[0].ix[0:]
    data2.ix[0:] =  data[1].ix[0:]
    data1.dropna(axis=0,how='all',inplace=True)
    data2.dropna(axis=0,how='all',inplace=True)
