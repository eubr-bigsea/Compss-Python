#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce import mergeReduce
from pycompss.api.api import compss_wait_on

import numpy as np
import math
import pandas as pd

def WorkloadBalancerOperation(df1, numFrag):
    """
    WorkloadBalancerOperation():

    Rebalance all the data in equal parts.

    :param data:       A list with numFrag pandas's dataframe;
    :param numFrag:    The number of fragments;
    :return:           Returns a balanced list with numFrag pandas's dataframe.
    """



    #first: check len of each frag
    len1 = [balancing_count( df1[f]) for f in range(numFrag)]
    len1 = compss_wait_on(len1)

    CV = np.std(len1) / np.mean(len1)
    #print "std:{} | mean:{} | CV:{}".format(np.std(len1),np.mean(len1),CV)
    if CV > 0.4:
        balanced = False
    else:
        balanced = True

    if not balanced:
        total  = max([total1,total2])
        size_p = int(math.ceil(float(total)/numFrag))

        for f in range(numFrag-1):
            partition_need1 = size_p - len1[f]
            #print "need1:{} ".format(partition_need1)
            if partition_need1>0:

                for g in xrange(f+1,numFrag):
                    if (len1[g] - partition_need1)<0:
                        off1 = len1[g]
                    else:
                        off1 = partition_need1

                    #print "off1:{}".format(off1)

                    df1[f] = balancing_f2_to_f1(df1[f], df1[g], off1)
                    #df1 = compss_wait_on(df1)
                    len1[g] -= off1
                    len1[f] += off1
                    partition_need1 -= off1

                    #print "MENOR | len2[f]:{}  | need2:{} | len2[g]:{} ".format(len1[f],len1[g],partition_need1)
                    #print "MENOR | len2[f]:{}  | need2:{} | len2[g]:{} ".format(len(df1[f]),len(df1[g]),partition_need1)
                        #print "STATUS:{}".format(len2)
                #print "Acabou com f:{} --> len:{}".format(f, len1[f])
            elif partition_need1<0:
                df1[f+1] = balancing_f1_to_f2(df1[f], df1[f+1], -partition_need1)
                #df1 = compss_wait_on(df1)
                len1[f+1] -= partition_need1
                len1[f] += partition_need1

                #print "MAIOR | len1[f]:{}  | need1:{} | len1[g]:{} ".format(len1[f],len1[f+1],partition_need1)
                #print "STATUS:{}".format(len2)

    return df1


@task(returns=int)
def balancing_count(df1):
    return len(df1)

@task( df_f1=INOUT, returns=list ) #df_f2=INOUT
def balancing_f1_to_f2(df_f1, df_f2, off1):
    #df_f1 MAIOR  --to-->    df_f2 MENOR

    tmp = df_f1.tail(off1)
    df_f1.drop(tmp.index, inplace=True)
    tmp.reset_index(drop=True,inplace=True)

    mynparray = df_f2.values
    mynparray = np.vstack((tmp,mynparray))
    df_f2 = pd.DataFrame(mynparray,columns = df_f2.columns)
    return df_f2


@task( df_f2=INOUT, returns=list ) #df_f2=INOUT
def balancing_f2_to_f1(df_f1, df_f2, off1):
    #df_f1 MAIOR  --to-->    df_f2 MENOR

    tmp = df_f2.head(off1)
    df_f2.drop(tmp.index, inplace=True)
    tmp.reset_index(drop=True,inplace=True)

    mynparray = df_f1.values
    mynparray = np.vstack((tmp,mynparray))
    df_f1 = pd.DataFrame(mynparray,columns = df_f1.columns)
    return df_f1
