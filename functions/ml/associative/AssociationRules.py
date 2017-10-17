#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from itertools import chain, combinations
from collections import defaultdict

from pycompss.api.parameter     import *
from pycompss.api.task          import task
from pycompss.functions.reduce  import mergeReduce




def AssociationRulesOperation(freqItems, settings):
    """
    AssociationRulesOperation():

    Generates the list of rules in the form: predecessor, successor
    and its confidence for each item passed by parameter.

    :param freqItems:     The pandas dataframe splitted in N parts;
    :param settings:      A dictionary with the informations:
      - 'col_item':       The column name of the items (default, items);
      - 'col_freq':       The column name of the support (default, support);
      - 'min_confidence': The minimum confidence (default, 0.5);
      - 'maxRules':       The maximum number of rules to return.
                          -1 to return all rules;
    :return               A list of pandas dataframe.
    """
    numFrag        = len(freqItems)
    col_item       = settings.get('col_item','items')
    col_freq       = settings.get('col_freq','support')
    min_confidence = settings.get('confidence', 0.5)
    maxRules       = int(settings.get('rules_count',-1))

    toRetRules = [ getRules(freqItems, col_item, col_freq, i, min_confidence ) for i in range(numFrag) ]

    if maxRules>-1:

        conf              = ['confidence']
        toRetRules_sorted = sort_byOddEven(toRetRules,conf ,numFrag)
        count             = [ count_transations(toRetRules_sorted[f]) for f in range(numFrag) ]
        mergedCount       = mergeReduce(mergecount, count)
        toRetRules        = [filterRules(toRetRules_sorted[f],mergedCount ,int(maxRules),f) for f in range(numFrag) ]

    return toRetRules

@task(returns=list)
def filterRules(toRetRules,count,maxRules,pos):
    print count
    total, partial = count
    if total > maxRules:
        gets = 0
        for i in range(pos):
            gets+=partial[i]
        number = maxRules-gets
        if number > partial[pos]: number = partial[pos]
        if number < 0:     number = 0
        return toRetRules.head(number)
    else:
        return toRetRules


@task(returns=list)
def getRules( freqItems, col_item, col_freq, i, minConfidence):

    toRetRules = []
    for index, row in freqItems[i].iterrows():
        item    = row[col_item]
        support = row[col_freq]
        if len(item)>0:
            _subsets = [list(x) for x in subsets(item)]

            for element in _subsets:
                remain = list(set(item).difference(element))

                if len(remain) > 0:
                    num = float(support)
                    den = getSupport(element,freqItems,col_item, col_freq)
                    confidence = num/den

                    if confidence > minConfidence:
                        r = [element, remain, confidence]
                        toRetRules.append(r)


    import pandas as pd
    rules = pd.DataFrame(toRetRules,columns=['Pre-Rule','Post-Rule','confidence'])

    return rules



def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def getSupport(element, freqItems,col_item, col_freq):
    for df in freqItems:
        for t,s  in zip(df[col_item].values, df[col_freq].values):
            if element == t:
                return s
    return float("inf")


@task(returns=list)
def mergeRules(rules1,rules2):
    return pd.concat([rules1,rules2])

@task(returns=list)
def count_transations(rules):
    return [len(rules), [len(rules)]]

@task(returns=list)
def mergecount(c1,c2):
    return [c1[0]+c2[0], c1[1]+c2[1]]

#----------------------

def sort_byOddEven(data,conf_col,numFrag):
    from pycompss.api.api import compss_wait_on
    for f in range(numFrag):
        data[f] = sort_p(data[f], conf_col)

    f = 0
    s = [ [] for i in range(numFrag/2)]

    nsorted = True
    while nsorted:
        if (f % 2 == 0):
            s = [ mergesort(data[i],data[i+1],conf_col) for i in range(numFrag)   if (i % 2 == 0)]
        else:
            s = [ mergesort(data[i],data[i+1],conf_col) for i in range(numFrag-1) if (i % 2 != 0)]

        s = compss_wait_on(s)

        if f>2:
            nsorted = any([ i ==-1 for i in s])
            #nsorted = False
        f +=1
    return data

@task(returns=list)
def sort_p(data, col):
    data.sort_values(col, ascending=[False], inplace=True)
    data = data.reset_index(drop=True)
    return data

@task(data1 = INOUT, data2 = INOUT, returns=int)
def mergesort(data1, data2, col):
    import pandas as pd
    """
    Returns 1 if [data1, data2] is sorted, otherwise is -1.
    """
    order = [False]
    n1 = len(data1)
    n2 = len(data2)

    if  n1 == 0 or n2 == 0:
        return 1

    idx_data1 = data1.index
    idx_data2 = data2.index
    j = 0
    k = 0
    nsorted = 1
    data = pd.DataFrame([],columns=data1.columns)
    t1 =  data1.ix[idx_data1[j]].values
    t2 =  data2.ix[idx_data2[k]].values
    for i in range(n1+n2):

        tmp = pd.DataFrame([t1,t2],columns=data1.columns)
        tmp.sort_values(col, ascending=order, inplace=True)
        idx = tmp.index

        if idx[0] == 1:
            nsorted = -1
            data.loc[i] = tmp.loc[1].values

            k+=1
            if k == n2:
                break
            t2 =  data2.ix[idx_data2[k]].values

        else:
            data.loc[i] = tmp.loc[0].values
            j+=1
            if j == n1:
                break
            t1 =  data1.ix[idx_data1[j]].values


    if k == n2:
        data = data.append(data1.ix[j:], ignore_index=True)
    else:
        data = data.append(data2.ix[k:], ignore_index=True)

    data1.ix[0:] = data.ix[:n1]
    data = data[data.index >= n1]
    data = data.reset_index(drop=True)
    data2.ix[0:] = data.ix[:]

    return  nsorted

#-----------------------------------
