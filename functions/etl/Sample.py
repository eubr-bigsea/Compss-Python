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
#   Sample

@task(returns=list)
def DefineNSample (total,value,seed,numFrag):

    if total[0] < value:
        value = total[0]
    np.random.seed(seed)
    ids = sorted(np.random.choice(total[0], value, replace=False))

    list_ids = [[] for i in range(numFrag)]

    frag = 0
    maxIdFrag = total[1][frag]
    oldmax = 0
    for i in ids:

        while i >= maxIdFrag:
            frag+=1
            oldmax = maxIdFrag
            maxIdFrag+= total[1][frag]

        list_ids[frag].append(i-oldmax)

    print "Total: {} |\nsize: {} |\nids: {} |\nlist_ids:{}".format(total,value,ids,list_ids)

    return list_ids

def Sample(data,params,numFrag):
    """
    Returns a sampled subset of this DataFrame.
    Parameters:
    - withReplacement -> can elements be sampled multiple times
                        (replaced when sampled out)
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen;
            fraction must be [0, 1]
        with replacement: expected number of times each element is chosen;
            fraction must be >= 0
    - seed -> seed for random operation.
    """

    indexes_split1 = [[] for i in range(numFrag)]
    if params["type"] == 'percent':
        percentage  = params['value']
        seed        = params['seed']
        indexes     = [ [] for i in range(numFrag)]
        partial_counts  = [CountRecord(data[i]) for i in range(numFrag)]
        total           = mergeReduce(mergeCount,partial_counts)
        indexes         = DefineSample(total,percentage,seed,numFrag)  # can be improved
        sample = [GetSample(data[i],indexes,i) for i in range(numFrag)]
        return sample
    elif params["type"] == 'value':
        value = params['value']
        seed = params['seed']
        partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
        total = mergeReduce(mergeCount,partial_counts)
        indexes_split1 = DefineNSample(total,value,seed,numFrag)
        indexes_split1 = compss_wait_on(indexes_split1,to_write = False)
        splits1 = [GetSample(data[i],indexes_split1[i]) for i in range(numFrag)]
        return splits1
    elif params['type'] == 'head':
        head = params['value']
        partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
        total = mergeReduce(mergeCount,partial_counts)
        total = compss_wait_on(total,to_write = False)
        sample = [GetHeadSample(data[i], total,i,head) for i in range(numFrag)]
        return sample


@task(returns=list)
def DefineSample(total,percentage,seed,numFrag):

    size = int(math.ceil(total[0]*percentage))

    np.random.seed(seed)
    ids = sorted(np.random.choice(total[0], size, replace=False))


    list_ids = [[] for i in range(numFrag)]
    frag = 0
    maxIdFrag = total[1][frag]
    oldmax = 0
    for i in ids:
        while i >= maxIdFrag:
            frag+=1
            oldmax = maxIdFrag
            maxIdFrag+= total[1][frag]
        list_ids[frag].append(i-oldmax)

    print "Total: {} |\nsize: {} |\nids: {} |\nlist_ids:{}".format(total,size,ids,list_ids)

    return list_ids




@task(returns=list)
def GetSample(data,indexes,i):
    indexes= indexes[i]
    sample = []
    print "DEGUG: GetSample"

    if len(data)>0:
        data = data.reset_index(drop=True)
        sample = data.loc[data.index.isin(indexes)]


    return sample
