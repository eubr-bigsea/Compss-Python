#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *

import numpy as np
import pandas as pd

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
        - operation: A dictionary with the functionst to be applied in
                     the aggregation:
            'mean':	    Computes the average of each group;
            'count':	Counts the total of records of each group;
            'first':	Returns the first element of group;
            'last':	    Returns the last element of group;
            'max':      Returns the max value of each group for one attribute;
            'min':	    Returns the min value of each group for one attribute;
            'sum':      Returns the sum of values of each group for one
                        attribute;
            'list':     Returns a list of objects with duplicates;
            'set':      Returns a set of objects with duplicate elements
                        eliminated.
    :param numFrag: The number of fragments;
    :return:        Returns a list with numFrag pandas's dataframe.

    example:
        settings['columns']   = ["col1"]
        settings['operation'] = {'col2':['sum'],'col3':['first','last']}
        settings['aliases']   = \
            {'col2':["Sum_col2"],'col3':['col_First','col_Last']}
    """


    columns   = params['columns']
    target    = params['aliases']
    operation = params['operation']


    data = [ aggregate_partial(data[f], columns, operation,  target)
                for f in range(numFrag)]

    ## buffer to store the join between each block
    import itertools
    buff =  list(itertools.combinations([x for x in range(numFrag)], 2))
    result = [[] for f in range(numFrag)]

    ## Merging the partial results

    def disjoint(a, b):
        return  set(a).isdisjoint(b)
    #
    # while len(buff)>0:
    #     step_list_i = []
    #     step_list_j = []
    #     step_list_i.append(buff[0][0])
    #     step_list_j.append(buff[0][1])
    #     del buff[0]
    #
    #     for i in range(len(buff)):
    #         tuples = buff[i]
    #         if  disjoint(tuples, step_list_i):
    #             if  disjoint(tuples, step_list_j):
    #                 step_list_i.append(tuples[0])
    #                 step_list_j.append(tuples[1])
    #                 del buff[i]
    #
    #     for x,y in zip(step_list_i,step_list_j):

    x_i = []
    y_i = []

    while len(buff)>0:
        x = buff[0][0]
        step_list_i = []
        step_list_j = []
        if x >= 0:
            y = buff[0][1]
            step_list_i.append(x)
            step_list_j.append(y)
            buff[0] = [-1,-1]
            for j in range( len(buff)):
                tuples = buff[j]
                if tuples[0] >=0:
                    if  disjoint(tuples, step_list_i):
                        if  disjoint(tuples, step_list_j):
                            step_list_i.append(tuples[0])
                            step_list_j.append(tuples[1])
                            buff[j] = [-1,-1]
        del buff[0]
        x_i.extend(step_list_i)
        y_i.extend(step_list_j)

    for x,y in zip(x_i,y_i):
        merge_aggregate(data[x], data[y], columns, operation, target)


    for f in range(numFrag):
        result[f] = remove_nan(data[f])


    return data

@task(returns=list)
def aggregate_partial(data, columns, operation, target):

    operation = replaceFunctionsName(operation)

    data = data.groupby(columns).agg(operation)

    newidx = []
    i = 0
    old=None
    for (n1,n2) in data.columns.ravel():

        if old != n1:
            old = n1
            i = 0

        newidx.append(target[n1][i])
        i+=1

    data.columns = newidx
    data = data.reset_index()

    return data


@task(data1=INOUT,data2=INOUT)
def merge_aggregate(data1, data2, columns, operation, target):
    """
        data1 and data2 will have at least, your currently size
    """

    n1 = len(data1)
    n2 = len(data2)

    operation = replaceNamebyFunctions(operation,target)

    data = pd.concat([data1,data2],axis=0, ignore_index=True)
    data = data.groupby(columns).agg(operation)

    #remove the diffent level
    data.reset_index(inplace=True)

    n = len(data)

    data1.columns = data.columns
    data2.columns = data.columns
    data1.ix[0:] = data.ix[:n1]

    data = data[data.index >= n1]
    data = data.reset_index(drop=True)
    data2.ix[0:] = data.ix[:]



def collectList(x):
    return  x.tolist()

def collectSet(x):
    # collectList and collectSet must be diferent functions,
    # otherwise pandas will raise error.
    return  x.tolist()

def mergeSet(series):
    return reduce(lambda x, y: list(set(x + y)), series.tolist())

def replaceFunctionsName(operation):
    # Replace 'set' and 'list' to the pointer of the real function
    for col in operation:
        for f in range(len(operation[col])):
            if operation[col][f] == 'list':
                operation[col][f] = collectList
            elif operation[col][f] == 'set':
                operation[col][f] = collectSet
    return operation

def replaceNamebyFunctions(operation,target):
    # Convert the operation dictionary to Alias ->  aggregate function
    new_operations = {}

    for col in operation:
        for f in range(len(operation[col])):
            if operation[col][f] == 'list':
                operation[col][f] = 'sum'
            elif operation[col][f] == 'set':
                operation[col][f] = mergeSet
            elif operation[col][f] == 'count':
                operation[col][f] = 'sum'

    for k in target:
        values = target[k]
        for i in range(len(values)):
            new_operations[values[i]] = operation[k][i]
    return new_operations


@task(returns=list)
def remove_nan(data):
    data.dropna(axis=0,how='all',inplace=True)
    return data
