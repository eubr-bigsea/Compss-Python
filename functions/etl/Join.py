#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd

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
        - 'case':      True is case-sensitive, otherwise is False
                       (default is True);
        - 'keep_keys': True to keep the keys of the second dataset,
                       (default, False).
        - 'suffixes':  Suffixes for attributes, a list with 2 values
                       (default, [_l,_r]);
    :param numFrag:    The number of fragments;
    :return:           Returns a list with numFrag pandas's dataframe.

    """


    result = [[] for i in range(numFrag)]

    key1 = params.get('key1',[])
    key2 = params.get('key2',[])
    TYPE = params.get('option','inner')

    if any([ len(key1) == 0,
             len(key2) == 0,
             len(key1) != len(key2),
             TYPE not in ['inner','left','right']
            ]):
        raise \
            Exception('You must inform the keys of first and second dataframe.'\
                'You also must inform the join type (inner,left or right join).')


    if TYPE == "inner":
        for i in range(numFrag):
            partial_join = \
                [InnerJoin(data1[i], data2[j], params) for j in range(numFrag)]
            result[i] = mergeReduce(mergeInnerJoin,partial_join)

    elif TYPE == "left":
        partial_m = [[] for i in range(numFrag)]
        for i in range(numFrag):
            partial_join = \
                [InnerJoin(data1[i],data2[j], params) for j in range(numFrag)]
            partial_m[i] = mergeReduce(mergeInnerJoin,partial_join)
            result[i]    = mergeLeftRightJoin(data1[i],partial_m[i],params)


    elif TYPE == "right":
        partial_m = [[] for i in range(numFrag)]
        for i in range(numFrag):
            partial_join = \
                [InnerJoin(data1[i],data2[j], params) for j in range(numFrag)]
            partial_m[i] = mergeReduce(mergeInnerJoin,partial_join)
            result[i]    = mergeLeftRightJoin(data2[i],partial_m[i], params)

    return result





def RenameCols(cols1,cols2,key,suf):
    convert = {}
    for c in range(len(cols1)):
        col = cols1[c]
        if col in cols2:
            n_col = "{}{}".format(col,suf)
            convert[col] = n_col
            key = [n_col if x==col  else x for x in key]
    return convert,key

@task(returns=list)
def InnerJoin(data1,data2,params):

    key1 = params['key1']
    key2 = params['key2']
    case_sensitive = params.get('case', True)
    keep = params.get('keep_keys', False)
    suffixes = params.get('suffixes',['_l','_r'])

    #Removing rows where NaN is in keys
    data1.dropna(axis=0, how='any', subset=key1, inplace=True)
    data2.dropna(axis=0, how='any', subset=key2, inplace=True)

    # Adding the suffixes before join.
    # This is necessary to preserve the keys
    # of the second table even though with equal name.
    cols1 = data1.columns
    cols2 = data2.columns
    LSuf = suffixes[0]
    RSuf = suffixes[1]

    convert1,key1 = RenameCols(cols1,cols2,key1,LSuf)
    convert2,key2 = RenameCols(cols2,cols1,key2,RSuf)


    data1.rename(columns=convert1, inplace=True)
    data2.rename(columns=convert2, inplace=True)

    #needed to left and right join operation
    if params['option'] == "right":
        data2['data1_InnerJoin'] = data2.index
    elif params['option'] == "left":
        data1['data1_InnerJoin'] = data1.index

    if case_sensitive:
        df_partial = pd.merge(  data1, data2, how='inner',
                                left_on=key1,right_on=key2
                                )

    else:
        # create a temporary copy of the two dataframe
        # with the keys in lower caption
        data1_tmp = data1[key1].applymap(lambda col: str(col).lower())
        data1_tmp['data1_tmp'] = data1_tmp.index
        data2_tmp = data2[key2].applymap(lambda col: str(col).lower())
        data2_tmp['data2_tmp'] = data2_tmp.index


        df_tmp = pd.merge(  data1_tmp, data2_tmp,
                            how='inner',left_on=key1, right_on=key2
                            )

        df_tmp.drop(key1+key2, axis=1, inplace=True)

        df_tmp  = pd.merge( data1, df_tmp,
                            left_index = True, right_on='data1_tmp'
                            )

        df_partial  = pd.merge( df_tmp, data2,
                                left_on='data2_tmp', right_index= True
                                )
        df_partial.drop(['data1_tmp','data2_tmp'], axis=1, inplace=True)


    if not keep:
        #remove all key columns of the second DataFrame
        df_partial.drop(key2, axis=1, inplace=True)

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

    key1 = params['key1']
    key2 = params['key2']
    case_sensitive = params.get('case', True)
    keep = params.get('keep_keys', False)
    suffixes = params.get('suffixes',['_l','_r'])

    cols1 = data1.columns # with original columns
    cols2 = data2.columns
    LSuf = suffixes[0]
    RSuf = suffixes[1]


    if params['option'] == "right":
        #Removing rows where NaN is in keys
        data1.dropna(axis=0, how='any', subset=key2, inplace=True)
        convert2 = {}
        for item in cols1:
            n_col = "{}{}".format(item,RSuf)
            if n_col  in cols2:
                convert2[item] = n_col
        data1.rename(columns=convert2, inplace=True)

    else:
        data1.dropna(axis=0, how='any', subset=key1, inplace=True)
        convert1 = {}
        for item in cols1:
            n_col = "{}{}".format(item,LSuf)
            if n_col in cols2:
                convert1[item] = n_col
        data1.rename(columns=convert1, inplace=True)

    #Remove rows which was joinned
    list_indexes = data2['data1_InnerJoin'].tolist()
    data1.drop(list_indexes, inplace=True)
    data2.drop('data1_InnerJoin', axis=1, inplace=True)

    #concatenating
    data = pd.concat([data1,data2])
    data.reset_index(drop=True,inplace=True)

    return data
