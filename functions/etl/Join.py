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
#   Join



def JoinOperation (data1,data2,params,numFrag):
    """
    JoinOperation():

    Joins with another DataFrame, using the given join expression.
    :param data1:      A list with numFrag pandas's dataframe;
    :param data2:      Other list with numFrag pandas's dataframe;
    :param params:     A dictionary that contains:
        - 'option':   'inner' to InnerJoin,
                      'left' to left join and 'right' to right join.
        - 'key1':      A list of keys of the first dataframe;
        - 'key2':      A list of keys of the second dataframe;
        - 'case':      True is case-sensitive, otherwise is False (default is True);
        - 'keep_keys': True to keep the keys of the second dataset (default is False).
    :param numFrag:    The number of fragments;
    :return:           Returns a list with numFrag pandas's dataframe.

    """



    result = [[] for i in range(numFrag)]
    key1 = params['key1']
    key2 = params['key2']

    TYPE = params['option']
    if TYPE == "inner":
        for i in range(numFrag):
            partial_join    = [ InnerJoin(data1[i], data2[j], params) for j in range(numFrag)]
            result[i]       = mergeReduce(mergeInnerJoin,partial_join)
        return result

    elif TYPE == "left":
        partial_m = [[] for i in range(numFrag)]
        for i in range(numFrag):
            partial_join    = [ InnerJoin(data1[i],data2[j], params) for j in range(numFrag) ]
            partial_m[i]    = mergeReduce(mergeInnerJoin,partial_join)
            result[i]       = mergeLeftRightJoin(data1[i],partial_m[i],params)
        return result

    elif TYPE] == "right":
        partial_m = [[] for i in range(numFrag)]
        for i in range(numFrag):
            partial_join    = [ InnerJoin(data1[i],data2[j], params) for j in range(numFrag) ]
            partial_m[i]    = mergeReduce(mergeInnerJoin,partial_join)
            result[i]       = mergeLeftRightJoin(data2[i],partial_m[i], params)
        return result

    else:
        return None




@task(returns=list)
def InnerJoin(data1,data2,params):

    key1 = params['key1']
    key2 = params['key2']
    case = params.get('case',True)
    keep = params.get('keep_keys',False)

    if params['option'] != "inner":
        data1['data1_InnerJoin'] = data1.index
    #data2.rename(columns={'ticket': 'ticket2'}, inplace=True) #only to test

    #data1 = data1[key1].apply(lambda col: col.str.lower())
    #data1['data1'] = data1.index
    #data2['data2'] = data2.index

    #data1.apply(lambda col: col.str.lower())

    if not case:
        data1_tmp = data1[key1].apply(lambda col: col.str.lower())
        data1_tmp['data1_tmp'] = data1_tmp.index
        data2_tmp = data2[key2].apply(lambda col: col.str.lower())
        data2_tmp['data2_tmp'] = data2_tmp.index

        df_tmp = pd.merge(data1_tmp, data2_tmp, how='inner',left_on=key1, right_on=key2)
        df_tmp.drop(key1+key2, axis=1, inplace=True)
        #print  df_tmp.head(10)
        #print "----------------"
        df_tmp  = pd.merge(data1,df_tmp, left_index = True, right_on='data1_tmp',suffixes=('', '_right'))
        #print df_tmp.head(10)
        #print "----------------"
        df_partial  = pd.merge(data2, df_tmp, left_index = True, right_on='data2_tmp',suffixes=('', '_right'))
        df_partial.drop(['data1_tmp','data2_tmp'], axis=1, inplace=True)
        #print df_partial

        if not keep:
            #finding for the keys of data2 where is not in data1
            needRemove2 = []
            for k in key2:
                if (k+"_right") in df_partial.columns:
                    needRemove2.append(k+"_right")
                elif (k in df_partial.columns) and k not in key1:
                    needRemove2.append(k)

            #print needRemove2
            df_partial.drop(needRemove2, axis=1, inplace=True)


    else:
        df_partial = pd.merge(data1, data2, how='inner',
                                                left_on=key1,
                                                right_on=key2,
                                                suffixes=('', '_right'))

        if not keep:
            #finding for the keys of data2 where is not in data1
            needKeep2 = [k for k in key2 if k not in key1]
            #print needKeep2
            df_partial.drop(needKeep2, axis=1, inplace=True)


    return df_partial


@task(returns=list)
def mergeInnerJoin(data1,data2):
    #in all the cases, columns will be correct
    if len(data1)>0:
        if len(data2)>0:
            return pd.concat([data1,data2])
        else:
            return data1
    else:
        return data2


@task(returns=list)
def mergeLeftRightJoin(data1, data2, params):



    # print data1.columns
    # print data2.columns
    # print "---"
    # print data1
    # print "---"
    # print data2
    if params['option'] == "right":
        key1 = params['key1']
        key2 = params['key2']
        cols2 = data2.columns
        cols1 = data1.columns

        convert ={}
        for c in range(len(cols1)):
            col = cols1[c]
            if ( (col+"_right") in cols2) and ((col+"_right") not in cols1):
                new = "{}_right".format(col)
                convert[col] = new
        data1.rename(columns=convert, inplace=True)

    list_indexes = data2['data1_InnerJoin'].tolist()
    #print list_indexes
    data1.drop(list_indexes, inplace=True)
    data2.drop('data1_InnerJoin', axis=1, inplace=True)

    data = pd.concat([data1,data2])
    #data =  pd.merge(data1, data2, how='left',left_on=key1, right_on=key2)

    return data

# @task(returns=list)
# def mergeRightJoin(data1,data2,id1,id2):
#
#     print data1
#     print id1
#     print "---"
#     print data2
#     print id2
#
#     if len(data1)>0:
#         if len(data2)>0:
#             data = data2.set_index(id2).merge(data1.set_index(id1))
#             print  data
#             return data
#         else:
#             return data1
#             #log = data2[1]
#     else:
#         return data2
