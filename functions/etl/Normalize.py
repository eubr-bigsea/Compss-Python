#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd

def NormalizeOperation(data, settings, numFrag):
    """
    NormalizeOperation():
    Perform a Feature scaling (Range Normalization) or a Standard Score
    Normalization on the selected columns;

    :param data:        A list with numFrag pandas's dataframe to perform
                        the Normalization.
    :param settings:    A dictionary that contains:
      - mode:
            * 'range' to perform the Range Normalization,
                also called Feature scaling. (default option)
            * 'standard' to perform the Standard Score Normalization.
      - attributes: 	A list of columns names to normalize;
      - alias:          Aliases of the new columns
                        (if empty, overwrite the old fields);
    :param numFrag:     A number of fragments;
    :return:            A list with numFrag pandas's dataframe
    """


    mode    = settings.get('mode','range')
    columns = settings.get('attributes',[])
    alias   = settings.get('alias',[])

    if len(columns) == 0:
        raise \
            Exception('You must insert the fields to perform the Normalization.')

    if mode == 'range':
        # generate a list of the min and the max element to each subset.
        minmax_partial = \
            [ aggregate_maxmin(data[f], columns) for f in range(numFrag)]
        # merge them into only one list
        minmax = mergeReduce(merge_maxmin, minmax_partial)
        # compute the new values to each subset
        result = [normalizate_byRange(data[f], columns, alias, minmax)
                    for f in range(numFrag)]

    elif mode == 'standard':
        # compute the sum of each subset column
        sum_partial = [aggregate_sum(data[f], columns) for f in range(numFrag)]
        # merge then to compute a mean
        mean = mergeReduce(merge_sum, sum_partial)
        # using this mean, compute the variance of each subset column
        sse_partial = \
                [aggregate_sse(data[f], columns, mean) for f in range(numFrag)]
        sse = mergeReduce(merge_sse, sse_partial)
        # finally, compute the new values to each subset
        result = [normalizate_byStandard(data[f], columns, alias, mean, sse)
                for f in range(numFrag)]

    else:
        raise Exception('Only supports range and standard Normalization types.')

    return result

# ------------------------
# By Range:

@task(returns=list)
def aggregate_maxmin(df, columns):
        #Generates a list of min and max values, excluding NaN values
        min_max_p = df[columns].describe().loc[['min','max']]
        return min_max_p.T.values.tolist()

@task(returns=list)
def merge_maxmin(minmax1,minmax2):
    minmax = []
    for di, dj in zip(minmax1,minmax2):
        minimum = di[0] if di[0] < dj[0] else dj[0]
        maximum = di[1] if di[1] > dj[1] else dj[1]
        minmax.append([minimum,maximum])
    return minmax

@task(returns=list)
def normalizate_byRange(data, columns, aliases, minmax):
    if len(aliases) != len(columns):
        aliases = columns

    for i, (alias,col) in  enumerate(zip(aliases,columns)):
        minimum, maximum = minmax[i]
        diff = maximum - minimum
        data[alias] = data[col].apply(lambda xi: float(xi - minimum)/diff)

    return data

# ------------------------
# By Standard Score Normalization:

@task(returns=list)
def aggregate_sum(df, columns):
        sum_p = df[columns].apply(lambda x: pd.Series([x.sum(),x.count()]))
        return sum_p.T.values.tolist()


@task(returns=list)
def merge_sum(sum1,sum2):
    sum_count = []
    for di, dj in zip(sum1,sum2):
        sum_t = di[0] + dj[0]
        count = di[1] + dj[1]
        sum_count.append([sum_t,count])
    return sum_count

@task(returns=list)
def aggregate_sse(df, columns, sum_count):
    sum_sse =  []

    for i,col in enumerate(columns):
        mi   = sum_count[i]
        mean = mi[0]/mi[1]
        sum_sse.append(df[col].apply(lambda xi: (xi-mean)**2 ).sum().T)
    return sum_sse


@task(returns=list)
def merge_sse(sum1,sum2):
    sum_count = []
    for di, dj in zip(sum1,sum2):
        sum_count.append(di+dj)
    return sum_count

@task(returns=list)
def normalizate_byStandard(data, columns, aliases, mean, sse):
    if len(aliases) != len(columns):
        aliases = columns

    for i, (alias,col) in  enumerate(zip(aliases,columns)):
        m   = mean[i][0]/mean[i][1]
        std = np.sqrt(sse[i]/ (mean[i][1])) #std population
        data[alias] = data[col].apply(lambda xi: float(xi - m)/std)

    return data
