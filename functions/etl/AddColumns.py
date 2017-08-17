#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks
from pycompss.api.api import compss_wait_on

import numpy as np
import pandas as pd
import math
import sys


def AddColumnsOperation(df1,df2,balanced,numFrag):
    """
        Function which add new columns in the pandas dataframe. The columns has
        to be in the same number of fragments

        :param list1: A dataframe already splited in numFrags.
        :param list2: A dataframe already splited in numFrags.
        :return: Returns a new dataframe.
    """
    if not balanced:
        df1,df2 = balancer(df1,df2,numFrag)


    result = [AddColumns_part(df1[f], df2[f]) for f in range(numFrag)]

    return result

@task(returns=list)
def AddColumns_part(a,b):
    #See more: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html

    if len(a)>0:
        if len(b)>0:
            a.reset_index(drop=True,inplace=True)
            b.reset_index(drop=True,inplace=True)
            return pd.merge(a, b, left_index=True, right_index=True, how='outer')
        else:
            return a
    else:
        return b




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
    print len1
    print len2

    if not balanced:
        total = max([total1,total2])
        size_p = int(math.ceil(float(total)/numFrag))

        for f in range(numFrag-1):
            need1 = size_p - len1[f]
            #print "need1:{} ".format(need1)
            if need1>0:
                for g in xrange(f+1,numFrag):
                    off1 = len1[g] - need1
                    #print "off1:{}".format(off1)
                    if off1>0:
                        df1[g] = balancing(df1[f], df1[g],off1)
                        len1[g] -= off1
                        need1   -= off1
                        len1[f] += off1
                        #print "MENOR | len1[g]:{}  | need1:{} | len1[f]:{} ".format(len1[g],need1,len1[f])
                        #print "STATUS:{}".format(len1)

            elif need1<0:
                df1[f+1] = balancing(df1[f], df1[f+1], -need1)
                len1[f+1] -= need1
                len1[f] += need1

                #print "MAIOR | len1[g]:{}  | need1:{} | len1[f]:{} ".format(len1[f+1],need1,len1[f])
                #print "STATUS:{}".format(len1)


        for f in range(numFrag-1):
            need2 = size_p - len2[f]
            #print "need1:{} ".format(need2)
            if need2>0:
                for g in xrange(f+1,numFrag):
                    off2 = len2[g] - need2
                    #print "off1:{}".format(off2)
                    if off1>0:
                        df2[g] = balancing(df2[f], df2[g],off2)
                        len2[g] -= off2
                        need2   -= off2
                        len2[f] += off2
                        #print "MENOR | len1[g]:{}  | need1:{} | len1[f]:{} ".format(len2[g],need2,len2[f])
                        #print "STATUS:{}".format(len2)

            elif need2<0:
                df2[f+1] = balancing(df2[f], df2[f+1], -need2)
                len2[f+1] -= need2
                len2[f] += need2

                #print "MAIOR | len1[g]:{}  | need1:{} | len1[f]:{} ".format(len2[f+1],need2,len2[f])
                #print "STATUS:{}".format(len2)
    return df1,df2


@task(returns=int)
def balancing_count(df1):
    return len(df1)

@task( df_f1=INOUT, returns=list ) #df_f2=INOUT
def balancing(df_f1, df_f2, off1):
    #df_f1 MAIOR  --to-->    df_f2 MENOR

    tmp = df_f1.tail(off1)
    df_f1.drop(tmp.index, inplace=True)
    tmp.reset_index(drop=True,inplace=True)

    mynparray = df_f2.values
    mynparray = np.vstack((tmp,mynparray))
    df_f2 = pd.DataFrame(mynparray,columns = df_f2.columns)
    return df_f2
