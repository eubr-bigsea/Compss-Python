#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.api.api import compss_wait_on

import numpy as np
import pandas as pd
import math
import sys


def AddColumnsOperation(df1,df2,balanced,sufixes,numFrag):
    """
        AddColumnsOperation():
        Merge two dataframes, column-wise, similar to the command
        paste in Linux.

        :param df1:         A list with numFrag pandas's dataframe;
        :param df2:         A list with numFrag pandas's dataframe;
        :param balanced:    True only if len(df1[i]) == len(df2[i]) to each i;
        :param numFrag:     The number of fragments;
        :param suffixes     Suffixes for attributes (a list with 2 values);
        :return:            Returns a list with numFrag pandas's dataframe.
    """
    if not balanced:
        df1, df2 = balancer(df1,df2,numFrag)


    result = [AddColumns_part(df1[f], df2[f],sufixes) for f in range(numFrag)]

    return result

@task(returns=list)
def AddColumns_part(a,b,suffixes):
    #See more: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html
    if len(suffixes) == 0:
        suffixes=('_x', '_y')
    a.reset_index(drop=True,inplace=True)
    b.reset_index(drop=True,inplace=True)
    return pd.merge(a, b, left_index=True, right_index=True,
                    how='outer', suffixes=suffixes)


#------------------------------------------------------------------------------



def balancer(df1,df2,numFrag):

    #first: check len of each frag
    len1 = [balancing_count( df1[f]) for f in range(numFrag)]
    len2 = [balancing_count( df2[f]) for f in range(numFrag)]
    len1 = compss_wait_on(len1)
    len2 = compss_wait_on(len2)
    total1 = sum(len1)
    total2 = sum(len2)

    balanced = True
    for i,j in zip(len1,len2):
        if i != j:
            balanced = False
            break

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

        for f in range(numFrag-1):
            partition_need2 = size_p - len2[f]
            #print "need2:{} ".format(partition_need2)
            if partition_need2>0:

                for g in xrange(f+1,numFrag):
                    if (len2[g] - partition_need2)<0:
                        off2 = len2[g]
                    else:
                        off2 = partition_need2

                    #print "off2:{}".format(off2)

                    df2[f] = balancing_f2_to_f1(df2[f], df2[g], off2)
                    #df2 = compss_wait_on(df2)
                    len2[g] -= off2
                    len2[f] += off2
                    partition_need2   -= off2

                    #print "MENOR | len2[f]:{}  | need2:{} | len2[g]:{} ".format(len2[f],len2[g],partition_need2)
                    #print "MENOR | len2[f]:{}  | need2:{} | len2[g]:{} ".format(len(df2[f]),len(df2[g]),partition_need2)
                        #print "STATUS:{}".format(len2)
                #print "Acabou com f:{} --> len:{}".format(f, len2[f])
            elif partition_need2<0:
                df2[f+1] = balancing_f1_to_f2(df2[f], df2[f+1], -partition_need2)
                #df2 = compss_wait_on(df2)
                len2[f+1] -= partition_need2
                len2[f] += partition_need2

                #print "MAIOR | len1[f]:{}  | need1:{} | len1[g]:{} ".format(len2[f],len2[f+1],partition_need2)
                #print "STATUS:{}".format(len2)

        #for i in df2:
        #    print len(i)

    return df1,df2


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
