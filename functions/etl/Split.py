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
#   Split

@task(returns=list)
def CountRecord(data):
    size = len(data)
    return [size,[size]]

@task(returns=list)
def mergeCount(data1,data2):
    return [data1[0]+data2[0],np.concatenate((data1[1], data2[1]), axis=0)]


@task(returns=list)
def DefineSplit (total,percentage,seed,numFrag):

    size_split1 = int(math.ceil(total[0]*percentage))

    np.random.seed(seed)
    ids = sorted(np.random.choice(total[0], size_split1, replace=False))
    # ids2 = [i for i in range(size_split1) if i not in ids1]
    #
    # ids = [ids1,ids2]
    # list_ids = [[] for i in range(numFrag)]
    # frag = 0
    # maxIdFrag = total[1][frag]
    # oldmax = 0
    # for i in ids:
    #     while i >= maxIdFrag:
    #         frag+=1
    #         oldmax = maxIdFrag
    #         maxIdFrag+= total[1][frag]
    #     list_ids[frag].append(i-oldmax)

    #print "Total: {} |\nsize_split1: {} |\nids: {} |\nlist_ids:{}".format(total,size_split1,ids,list_ids)

    return ids

@task(returns=list)
def GetSplit1(data,indexes_split1):
    split1 = []
    print "DEGUG: GetSplit1"

    if len(data)>0:
        print data
        print data.index
        print "List of index: %s" % indexes_split1

        #df.loc[~df.index.isin(t)]
        split1 = data.loc[data.index.isin(indexes_split1)]


    #     if
    # pos= 0
    # if len(indexes_split1)>0:
    #     for i  in range(len(data)):
    #         if i == indexes_split1[pos]:
    #             split1.append(data[i])
    #             if pos < (len(indexes_split1)-1):
    #                 pos+=1

    print split1
    return split1

@task(returns=list)
def GetSplit2(data,indexes_split2):
    print "DEGUG: GetSplit2"

    split2 = []
    pos= 0

    if len(data)>0:
        print data
        print data.index
        print "List of index: %s" % indexes_split2

        split2 =data.loc[~data.index.isin(indexes_split2)]


    print split2
    return split2


def Split(data,settings,numFrag):
    percentage = settings['percentage']
    seed = settings['seed']

    partial_counts = [CountRecord(data[i]) for i in range(numFrag)]
    total = mergeReduce(mergeCount,partial_counts) #Remove this
    indexes = DefineSplit(total,percentage,seed,numFrag)
    splits1 = [GetSplit1(data[i],indexes) for i in range(numFrag)]
    splits2 = [GetSplit2(data[i],indexes) for i in range(numFrag)]

    return  [splits1,splits2]
