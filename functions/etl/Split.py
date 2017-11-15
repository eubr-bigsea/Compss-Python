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

def SplitOperation(data,settings,numFrag):
    """
    SplitOperation():
    Randomly splits a Data Frame into two data frames.

    :param data:      A list with numFrag pandas's dataframe;
    :settings:        A dictionary that contains:
      - 'percentage': Percentage to split the data (default, 0.5);
      - 'seed':       Optional, seed in case of deterministic random operation.
    :return:          Returns two lists with numFrag pandas's dataframe with
                      distincts subsets of the input.

    Note:   if percentage = 0.25, the final dataframes
            will have respectively,25% and 75%.
    """

    percentage = settings.get('percentage', 0.5)
    seed = settings.get('seed',None)

    if percentage < 0 or percentage > 1:
        raise Exception("Please inform a valid percentage [0, 1].")

    # count the size of each fragment and create a mapping
    # of the elements to be selected.
    partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
    total = mergeReduce(mergeCount,partial_counts)
    indexes = DefineSplit(total,percentage,seed,numFrag)

    splits1 = [ []  for i in range(numFrag)]
    splits2 = [ []  for i in range(numFrag)]
    for i in range(numFrag):
        splits1[i] = GetSplits(data[i],indexes,True,i)
        splits2[i] = GetSplits(data[i],indexes,False,i)

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
    ids = np.array(sorted(np.random.choice(total, size, replace=False)))

    n_list = np.cumsum(n_list)
    list_ids = [[] for i in range(numFrag)]

    first_id = 0
    for i in range(numFrag):
        last_id = n_list[i]
        idx = (ids >= first_id) & (ids < last_id)
        print idx
        list_ids[i] =  ids[idx]
        first_id = last_id

    # frag = 0
    # maxIdFrag = n_list[frag]
    # oldmax = 0
    # for i in ids:
    #     while i >= maxIdFrag:
    #         frag+=1
    #         oldmax = maxIdFrag
    #         maxIdFrag+= n_list[frag]
    #
    #     list_ids[frag].append(i-oldmax)

    return list_ids

@task(returns=list)
def GetSplits(data,indexes,part1,frag):

    data = data.reset_index(drop=True)

    if part1:
        split = data.loc[data.index.isin(indexes[frag])]
    else:
        split = data.loc[~data.index.isin(indexes[frag])]

    return split
