#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"



from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.api.api import compss_wait_on

import numpy as np
import pandas as pd
import math


def SortOperation(data,settings,numFrag):
    """
        SortOperation():

        Returns a new DataFrame sorted by the specified column(s).
        :param data:        A list with numFrag pandas's dataframe;
        :param settings:    A dictionary that contains:
            - algorithm:
                * 'odd-even', to sort using Odd-Even Sort;
                * 'bitonic',  to sort using Bitonic Sort (only if numFrag is power of 2);
            - columns:      The list of columns to be sorted.
            - ascending:    A list indicating whether the sort order is ascending (True) for the columns.
        :param numFrag:     The number of fragments;
        :return:            A list with numFrag pandas's dataframe

        Condition:  the list of columns should have the same size of the list
                    of boolean to indicating if it is ascending sorting.
    """


    def is_power2(num):
        return ((num & (num - 1)) == 0) and num != 0

    algorithm = settings.get('algorithm','odd-even')

    if algorithm == "bitonic" and is_power2(numFrag):
        data = sort_byBittonic(data,settings,numFrag)
    else:
        data = sort_byOddEven(data,settings,numFrag)


    return data

def sort_byBittonic(data,settings,numFrag):
    """
    Given an unordered sequence of size 2*fragments, exactly
    log2 (fragments) stages of merging are required to produce
    a completely ordered list.
    """

    for f in range(numFrag):
        data[f] = sort_p(data[f], settings)

    step   = 1
    stage  = 1
    gap    = 2
    max_steps = int(np.log2(numFrag))
    while step <= max_steps:
        if (stage <= step) and (stage>0):
            f   = 0
            if stage == 1:
                gap = 1
                while f<numFrag:
                    #print "[INFO] | stage == 1 | step:{} | stage{}| idx_{} and idx_{} | gap: {}".format(step,stage,f,f+gap,gap)
                    mergesort(data[f],data[f+gap], settings)
                    f+=(gap+1)

            elif stage == step:
                gap =  2**step -1
                while gap>=1:
                    #print "[INFO] | stage == step  step:{} | stage{}| idx_{} and idx_{} | gap: {}".format(step,stage,f,f+gap,gap)
                    mergesort(data[f],data[f+gap], settings)
                    f+=1
                    gap-=2

            else:
                gap = gap/2
                while f<numFrag:
                    #print "[INFO] | stage != step |  step:{} | stage{}| idx_{} and idx_{} | gap: {}".format(step,stage,f,f+gap,gap)
                    mergesort(data[f],data[f+gap], settings)
                    f+=(gap+1)
            stage-=1

        else:
            step+=1
            stage=step

    return data

# @task(data1=INOUT, data2=INOUT)
# def bitonic_sort(data1,data2,settings):
#     col = settings['columns']
#     order = settings['ascending']
#     n1 = len(data1)
#     n2 = len(data2)
#     data = pd.DataFrame([],columns=data1.columns)
#
#     data = pd.concat([data1,data2],axis=0, ignore_index=True)
#     data.sort_values(col, ascending=order,inplace=True)
#
#     data = data.reset_index(drop=True)
#     data1.ix[0:] = data.ix[:n1]
#     data = data[data.index >= n1]
#     data = data.reset_index(drop=True)
#     data2.ix[0:] = data.ix[0:]


def sort_byOddEven(data,settings,numFrag):
    for f in range(numFrag):
        data[f] = sort_p(data[f], settings)

    f = 0
    s = [ [] for i in range(numFrag/2)]

    nsorted = True
    while nsorted:
        if (f % 2 == 0):
            s = [ mergesort(data[i],data[i+1],settings) for i in range(numFrag)   if (i % 2 == 0)]
        else:
            s = [ mergesort(data[i],data[i+1],settings) for i in range(numFrag-1) if (i % 2 != 0)]

        s = compss_wait_on(s)

        if f>2:
            nsorted = any([ i ==-1 for i in s])
            #nsorted = False
        f +=1
    return data





@task(returns=list)
def sort_p(data, settings):
    col = settings['columns']
    order = settings['ascending']
    data.sort_values(col, ascending=order, inplace=True)
    data = data.reset_index(drop=True)
    return data

@task(data1 = INOUT, data2 = INOUT, returns=int)
def mergesort(data1, data2, settings):
    """
    Returns 1 if [data1, data2] is sorted, otherwise is -1.
    """
    col   = settings['columns']
    order = settings['ascending']
    n1 = len(data1)
    n2 = len(data2)
    
    if  n1 == 0 or n2 == 0:
        return 1

    idx_data1 = data1.index
    idx_data2 = data2.index
    j = 0
    k = 0
    nsorted = 1
    data = pd.DataFrame([],columns=data1.columns)
    t1 =  data1.ix[idx_data1[j]].values
    t2 =  data2.ix[idx_data2[k]].values
    for i in range(n1+n2):

        tmp = pd.DataFrame([t1,t2],columns=data1.columns)
        tmp.sort_values(col, ascending=order, inplace=True)
        idx = tmp.index

        if idx[0] == 1:
            nsorted = -1
            data.loc[i] = tmp.loc[1].values

            k+=1
            if k == n2:
                break
            t2 =  data2.ix[idx_data2[k]].values

        else:
            data.loc[i] = tmp.loc[0].values
            j+=1
            if j == n1:
                break
            t1 =  data1.ix[idx_data1[j]].values


    if k == n2:
        data = data.append(data1.ix[j:], ignore_index=True)
    else:
        data = data.append(data2.ix[k:], ignore_index=True)

    data1.ix[0:] = data.ix[:n1]
    data = data[data.index >= n1]
    data = data.reset_index(drop=True)
    data2.ix[0:] = data.ix[:]

    return  nsorted
