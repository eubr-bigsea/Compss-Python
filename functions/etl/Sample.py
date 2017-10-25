#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

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
            * 'percent':   Sample a random amount of records (default)
            * 'value':     Sample a N random records
            * 'head':      Sample the N firsts records of the dataframe
        - seed :           Optional, seed for the random operation.
        - int_value:       Value N to be sampled (in 'value' or 'head' type)
        - per_value:       Percentage to be sampled (in 'value' or 'head' type)
    :param numFrag:        The number of fragments;
    :return:               A list with numFrag pandas's dataframe.

    """

    value, int_per = Validate(params)
    TYPE = params.get("type", 'percent')

    #Its necessary count the distribuition of the data in each fragment
    partial_counts  = [CountRecord(data[i]) for i in range(numFrag)]
    N_list          = mergeReduce(mergeCount,partial_counts)

    if TYPE  == 'percent':
        seed        = params.get('seed', None)
        indexes     = DefineNSample(N_list,None,seed,True,'null',numFrag)
        data = [GetSample(data[i],indexes,i) for i in range(numFrag)]
    elif TYPE  == 'value':
        #value       = params['value']
        seed        = params.get('seed', None)
        indexes     = DefineNSample(N_list,value,seed,False,int_per,numFrag)
        data = [GetSample(data[i],indexes,i) for i in range(numFrag)]
    elif TYPE == 'head':
        #head    = params['value']
        indexes = DefineHeadSample(N_list,value,int_per,numFrag)
        data = [GetSample(data[i],indexes,i) for i in range(numFrag)]

    return data



def Validate(params):
    TYPE = params.get("type", 'percent')
    if TYPE not in ['percent','value','head']:
        raise Exception('You must inform a valid sampling type.')

    value = -1
    op = 'int'
    if TYPE == 'head' or TYPE == 'value':
        if  'int_value'  in params:
            value = params['int_value']
            op = 'int'
            if not isinstance(value, int) and value < 0:
                raise Exception('`int_value` must be a positive integer.')
        elif 'per_value' in params:
            value = params['per_value']
            op = 'per'
            if value > 1 or value < 0:
                raise Exception('Percentage value must between 0 and 1.0.')
        else:
            raise Exception('Using `Head` or `value` sampling type you need to '
                            'set `int_value` or `per_value` setting as well.')

    return value, op

@task(returns=list)
def CountRecord(data):
    size = len(data)
    return [size,[size]]

@task(returns=list)
def mergeCount(data1,data2):
    return [data1[0]+data2[0], np.concatenate((data1[1], data2[1]), axis=0)]

@task(returns=list)
def DefineNSample (N_list,value,seed,random,int_per,numFrag):

    total, n_list = N_list
    if int_per == 'int':
        if total < value:
            value = total
    elif int_per == 'per':
        value = int(math.ceil(total*value))


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
def DefineHeadSample (N_list,head,int_per,numFrag):

    total, n_list = N_list

    if int_per == 'int':
        if total < head:
            head = total
    elif int_per == 'per':
        head = int(math.ceil(total*head))

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
