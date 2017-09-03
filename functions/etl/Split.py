#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd
import math

def SplitOperation(data,settings,numFrag):
    """
    SplitOperation():
    Randomly splits a Data Frame into two data frames.

    :param data:      A list with numFrag pandas's dataframe;
    :settings:        A dictionary that contains:
      - 'percentage': Percentage to split the data;
      - 'seed':       Optional, seed in case of deterministic random operation.
    :return:          Returns two lists with numFrag pandas's dataframe with
                      distincts subsets of the input.

    """

    percentage = settings.get('percentage',0)
    seed = settings.get('seed',None)

    partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
    total = mergeReduce(mergeCount,partial_counts)
    indexes = DefineSplit(total,percentage,seed,numFrag)

    splits1 = [GetSplits(data[i],indexes,True,i)  for i in range(numFrag)]
    splits2 = [GetSplits(data[i],indexes,False,i) for i in range(numFrag)]

    return  [splits1, splits2]


@task(returns=list)
def CountRecord(data):
    size = len(data)
    return [size,[size]]

@task(returns=list)
def mergeCount(data1,data2):
    return [data1[0]+data2[0],np.concatenate((data1[1], data2[1]), axis=0)]


@task(returns=list)
def DefineSplit (N_list,percentage,seed,numFrag):
    total, n_list = N_list
    size = int(math.floor(total*percentage))

    np.random.seed(seed)
    ids = sorted(np.random.choice(total, size, replace=False))

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
def GetSplits(data,indexes,part1,frag):

    data = data.reset_index(drop=True)

    if part1:
        split = data.loc[data.index.isin(indexes[frag])]
    else:
        split = data.loc[~data.index.isin(indexes[frag])]

    return split
