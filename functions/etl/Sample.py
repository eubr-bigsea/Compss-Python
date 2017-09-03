#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd
import math


def SampleOperation(data,params,numFrag):
    """
    SampleOperation():

    Returns a sampled subset of the input panda's dataFrame.
    :param data:           A list with numFrag pandas's dataframe;
    :param params:         A dictionary that contains:
        - type:
            * 'percent':   Sample a random amount of records
            * 'value':     Sample a N random records
            * 'head':      Sample the N firsts records of the dataframe
        - seed :           Optional, seed for the random operation.
        - value:           Value N to be sampled (in 'value' or 'head' type)
    :param numFrag:        The number of fragments;
    :return:               A list with numFrag pandas's dataframe.

    """

    partial_counts  = [CountRecord(data[i]) for i in range(numFrag)]
    N_list          = mergeReduce(mergeCount,partial_counts)

    TYPE = params.get("type", None)

    if TYPE  == 'percent':
        seed        = params.get('seed', None)
        indexes     = DefineNSample(N_list,None,seed,True,numFrag)
        data = [GetSample(data[i],indexes,i) for i in range(numFrag)]
    elif TYPE  == 'value':
        value       = params.get('value', 0)
        seed        = params.get('seed', None)
        indexes     = DefineNSample(N_list,value,seed,False,numFrag)
        data = [GetSample(data[i],indexes,i) for i in range(numFrag)]
    elif TYPE == 'head':
        head    = params.get('value', 0)
        indexes = DefineHeadSample(N_list,head,numFrag)
        data = [GetSample(data[i],indexes,i) for i in range(numFrag)]

    return data

@task(returns=list)
def CountRecord(data):
    size = len(data)
    return [size,[size]]

@task(returns=list)
def mergeCount(data1,data2):
    return [data1[0]+data2[0], np.concatenate((data1[1], data2[1]), axis=0)]

@task(returns=list)
def DefineNSample (N_list,value,seed,random,numFrag):

    total, n_list = N_list
    if total < value:
        value = total

    if random:
        np.random.seed(seed)
        percentage = np.random.random_sample()
        value = int(math.ceil(total*percentage))


    np.random.seed(seed)
    ids = sorted(np.random.choice(total, value, replace=False))

    list_ids = [[] for i in range(numFrag)]

    frag = 0
    maxIdFrag = n_list[frag]
    oldmax = 0
    for i in ids:
        while i >= maxIdFrag:
            frag+=1
            oldmax = maxIdFrag
            maxIdFrag+= n_list[frag]

        list_ids[frag].append(i-oldmax)

    return list_ids

@task(returns=list)
def DefineHeadSample (N_list,head,numFrag):

    total, n_list = N_list

    if total < head:
        head = total

    list_ids = [[] for i in range(numFrag)]

    frag = 0
    while head > 0:
        off = head - n_list[frag]
        if off < 0:
            off = head
        else:
            off = n_list[frag]

        list_ids[frag] = [i for i in range(off)]
        head -= off
        frag+=1

    return list_ids


@task(returns=list)
def GetSample(data,indexes,i):
    indexes = indexes[i]
    data = data.reset_index(drop=True)
    sample = data.loc[data.index.isin(indexes)]

    return sample
