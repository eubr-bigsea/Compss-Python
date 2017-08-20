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



def JoinOperation (data1,data2,id1,id2,params,numFrag):
    result = [[] for i in range(numFrag)]

    if params['option'] == "inner":
        for i in range(numFrag):
            partial_join    = [InnerJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            result[i]  = mergeReduce(mergeInnerJoin,partial_join)
        return result
            #merged_join[i]  = merged_join[i][0]       ------------- TEm como enviar 2 ou mais ##############

    elif params['option'] == "left":
        partial_m = [[] for i in range(numFrag)]
        for i in range(numFrag):
            partial_join    = [InnerJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            partial_m[i]    = mergeReduce(mergeInnerJoin,partial_join)
            result[i]       = mergeLeftJoin(partial_m[i],data1[i],id2,id1)
        return result
    elif params['option'] == "right":
        partial_m = [[] for i in range(numFrag)]
        for i in range(numFrag):
            partial_join    = [InnerJoin(data1[i],data2[j],id1,id2) for j in range(numFrag)]
            partial_m[i]    = mergeReduce(mergeInnerJoin,partial_join)
            result[i]       = mergeRightJoin(partial_m[i],data2[i],id1,id2)
        return result




@task(returns=list)
def InnerJoin(data1,data2,id1,id2):

    print data1
    print "----"
    print data2

    if len(data1)>0 and len(data2)>0:
            df_partial = pd.merge(data1,data2, how='inner', left_on=id1, right_on=id2)
            print df_partial
            return df_partial
    else:
            return []

    return df_partial

@task(returns=list)
def LeftJoin(data1,data2,id1,id2):

    print data1
    print "----"
    print data2

    if len(data1)>0 :
        if len(data2)>0 :

            df_partial = pd.merge(data1,data2, how='left', left_on=id1, right_on=id2,indicator=True)
            idx = list(set(id1+id2))
            df_partial.set_index(id1)
            print df_partial
            return df_partial
        else:
            df_partial = data1
            print df_partial
            return df_partial
    else:
        df_partial = data2
        print df_partial
        return df_partial


@task(returns=list)
def RightJoin(data1,data2,id1,id2):
    L = len(data2)

    if L ==0:
        return [[],[]]
    if len(data1) == 0:
        return [[],[]]

    C = len(data1[0]) + len(data2[0]) - len(id1)
    size = len(data1[0]) - len(id1)

    #print "L :{} | C (A+B) : {} |  sub(A-B): {}".format(L,C,size)

    b = np.zeros((L,C ))
    b[:,size:] = data2
    log = np.zeros((L,1 ))

    print b

    for i in range(len(data2)):
        for j in range(len(data1)):
            found=True

            for i1,j2 in  zip(id1,id2):
                if data1[i][i1] != data2[j][j2]:
                    found=False
            if found:
                sub = np.delete(data1[j], id2, None)

                b[i][:size] =  sub
                log[i]=1


    print [b,log]

    return [b,log]



@task(returns=list)
def mergeInnerJoin(data1,data2):

    if len(data1)>0:
        if len(data2)>0:
            return pd.concat([data1,data2])
        else:
            return data1
    else:
        return data2

@task(returns=list)
def mergeLeftJoin(data1,data2,id1,id2):

    print data1
    print "---"
    print data2

    if len(data1)>0:
        if len(data2)>0:
            data = data1.set_index(id1).merge(data2.set_index(id2))
            print  data
            return data
        else:
            return data1
            #log = data2[1]
    else:
        return data2

@task(returns=list)
def mergeRightJoin(data1,data2,id1,id2):

    print data1
    print id1
    print "---"
    print data2
    print id2

    if len(data1)>0:
        if len(data2)>0:
            data = data2.set_index(id2).merge(data1.set_index(id1))
            print  data
            return data
        else:
            return data1
            #log = data2[1]
    else:
        return data2
