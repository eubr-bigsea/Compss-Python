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
            - operation: A dictionary with the functionst to be applied in the aggregation:
                'mean':	    Computes the average of each group;
                'count':	Counts the total of records of each group;
                'first':	Returns the first element of group;
                'last':	    Returns the last element of group;
                'max':      Returns the max value of each group for one attribute;
                'min':	    Returns the min value of each group for one attribute;
                'sum':      Returns the sum of values of each group for one attribute;
                'list':     Returns a list of objects with duplicates;
                'set':      Returns a set of objects with duplicate elements eliminated.
        :param numFrag: The number of fragments;
        :return:        Returns a list with numFrag pandas's dataframe.

        example:
            settings['columns']   = ["col1"]
            settings['operation'] = {'col2':['sum'],'col3':['first','last']}
            settings['aliases']   = {'col2':["Sum_col2"],'col3':['col_First','col_Last']}
    """


    columns   = params['columns']
    target    = params['aliases']
    operation = params['operation']


    data = [ aggregate_partial(data[f], columns, operation,  target) for f in range(numFrag)]

    ## buffer to store the join between each block
    buff = [(f,g) for f in range(numFrag)  for g in xrange(f,numFrag) if f != g]


    ## Merging the partial results

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
            merge_aggregate(data[x], data[y], columns, operation, target)



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


@task(data1=INOUT, data2=INOUT)
def merge_aggregate(data1, data2, columns, operation, target):

    operation = replaceNamebyFunctions(operation,target)

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

    for k in target:
        values = target[k]
        for i in range(len(values)):
            new_operations[values[i]] = operation[k][i]
    return new_operations
