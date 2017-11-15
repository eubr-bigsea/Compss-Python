#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce import mergeReduce

import numpy  as np
import pandas as pd

def WorkloadBalancerOperation(df1, forced, numFrag):
    """
    WorkloadBalancerOperation():

    Redistribute the data in equal parts if it's unbalanced. It is considered
    an unbalanced dataframe if the coefficient of variation (CV) between
    fragments is greater than 0.20.

    :param data:       A list with numFrag pandas's dataframe;
    :param forced:     True to force redistribution of data,
                       False to use heuristic based on the CV;
    :param numFrag:    The number of fragments;
    :return:           Returns a balanced list with numFrag pandas's dataframe.
    """

    from pycompss.api.api import compss_wait_on

    #first: check the distribution of the data
    len1 = [balancing_count( df1[f]) for f in range(numFrag)]
    len1 = mergeReduce(mergeCount,len1)
    len1 = compss_wait_on(len1)
    total = len1[0]
    len1 = len1[1]

    if forced:
        balanced = False
    else:
        CV = np.std(len1) / np.mean(len1)
        print "Coefficient of variation:{}".format(CV)
        if CV > 0.20:
            balanced = False
            print 'It assumed that do not compensates distribute the database.'
        else:
            balanced = True
            print 'It assumed that compensates distribute the database.'


    if not balanced:

        equal_size = int(np.ceil(float(total)/numFrag))

        for f in range(numFrag-1):
            quantity_needed = equal_size - len1[f]

            if quantity_needed>0:
                #If lines are missing, get from the next fragments
                for g in xrange(f+1,numFrag):
                    # amount that the next fragment can yield
                    if len1[g] >= quantity_needed:
                        offset = quantity_needed
                    else:
                        offset = len1[g]

                    df1[f] = balancing_f2_to_f1(df1[f], df1[g], offset)
                    len1[g] -= offset
                    len1[f] += offset
                    quantity_needed -= offset

            elif quantity_needed<0:
                #if lines are in excess, move it to the next block
                df1[f+1] = balancing_f1_to_f2(df1[f],df1[f+1], -quantity_needed)
                len1[f+1] -= quantity_needed
                len1[f]   += quantity_needed


    return df1


@task(returns=list)
def balancing_count(df1):
    return [len(df1), [len(df1)]]

@task(returns=list)
def mergeCount(len1,len2):
    return [len1[0]+len2[0], len1[1]+len2[1] ]

@task( df_f1=INOUT, returns=list ) #df_f2=INOUT
def balancing_f1_to_f2(df_f1, df_f2, off1):
    # Get the tail offset lines from df_f1
    # and put at the head of df_f2

    tmp = df_f1.tail(off1)
    df_f1.drop(tmp.index, inplace=True)
    tmp.reset_index(drop=True,inplace=True)

    mynparray = df_f2.values
    mynparray = np.vstack((tmp,mynparray))
    df_f2 = pd.DataFrame(mynparray,columns = df_f2.columns)
    return df_f2


@task( df_f2=INOUT, returns=list ) #df_f2=INOUT
def balancing_f2_to_f1(df_f1, df_f2, offset):
    # Get the head offset lines from df_f2
    # and put at the tail of df_f1
    tmp = df_f2.head(offset)
    df_f2.drop(tmp.index, inplace=True)
    tmp.reset_index(drop=True,inplace=True)

    mynparray = df_f1.values
    mynparray = np.vstack((mynparray,tmp))
    df_f1 = pd.DataFrame(mynparray,columns = df_f1.columns)
    return df_f1
