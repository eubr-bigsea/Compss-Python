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
        AggregationOperation():

        Computes aggregates and returns the result as a DataFrame.

        :param data:     A list with numFrag pandas's dataframe;
        :param params:   A dictionary that contains:
            - columns:   A list with the columns names to aggregates;
            - alias:     A dictionary with the aliases of all aggregated columns;
            - operation: A dictionary with the functionst to be applied in the aggregation:
                'mean':	    Computes the average of each group;
                'count':	Counts the total of records of each group;
                'first':	Returns the first element of group;
                'last':	    Returns the last element of group;
                'max':      Returns the max value of each group for one attribute;
                'min':	    Returns the min value of each group for one attribute;
                'sum':      Returns the sum of values of each group for one attribute;
        :param numFrag: The number of fragments;
        :return:        Returns a list with numFrag pandas's dataframe.

        example:
            settings['columns'] = ["col1"]
            settings['operation'] = {'col2':['sum'],'col3':['first','last']}
            settings['aliases']   = {'col2':["Sum_col2"],'col3':['col_First','col_Last']}
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
