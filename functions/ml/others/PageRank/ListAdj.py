#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import sys
sys.path.insert(0, '/home/lucasmsp/workspace/BigSea/Compss-Python/functions/data')
from data_functions import *



def create_AdjList(data,numFrag):
    from pycompss.api.api import compss_wait_on
    inlinks  = [findInlinks(data[i]) for i in range(numFrag)]
    inlinks  = mergeReduce( MergeInlinks,inlinks)
    inlinks  = Partitionize(inlinks,numFrag)
    inlinks = compss_wait_on(inlinks)

    partial  = [create_AdjList_partial(data[j],inlinks[i]) for i in range(numFrag) for j in range(numFrag)]
    merged   = [mergeReduce(MergeInlinks,partial[i])  for i in range(numFrag)]

    return merged

@task(returns=list)
def findInlinks(data):
    inlinks = []
    for row in data:
        if row[0] not in inlinks:
            inlinks.append(row[0])
    return inlinks

@task(returns=list)
def MergeInlinks(data1,data2):
    return data1 + data2


@task(returns=dict)
def create_AdjList_partial(data,inlinks):
    dict_partial = {}
    for link in inlinks:
        dict_partial[link] = []

    for row in data:
        if row[0] in dict_partial:
            dict_partial[row[0]] = dict_partial[row[0]] + row[1:]

    return dict_partial

@task(returns=dict)
def merge_AdjList(data1,data2):
    print data1
    print data2
    for entry ,value in data2.iteritems():
        if entry in data1:
            data1[entry] += value
        else:
            data1[entry] = value
    return data1




#
# def create_AdjList(data,numFrag):
#     from pycompss.api.api import compss_wait_on
#     inlinks  = [findInlinks(data[i]) for i in range(numFrag)]
#     inlinks  = mergeReduce( MergeInlinks,inlinks)
#     inlinks  = Partitionize(inlinks,numFrag)
#     inlinks = compss_wait_on(inlinks)
#
#     partial  = [create_AdjList_partial(data[j],inlinks[i]) for i in range(numFrag) for j in range(numFrag)]
#     merged   = [mergeReduce(MergeInlinks,partial[i])  for i in range(numFrag)]
#
#     return merged
#
# @task(returns=list)
# def findInlinks(data):
#     inlinks = []
#     for row in data:
#         if row[0] not in inlinks:
#             inlinks.append(row[0])
#     return inlinks
#
# @task(returns=list)
# def MergeInlinks(data1,data2):
#     return data1 + data2
#
#
# @task(returns=dict)
# def create_AdjList_partial(data,inlinks):
#     dict_partial = {}
#     for link in inlinks:
#         dict_partial[link] = []
#
#     for row in data:
#         if row[0] in dict_partial:
#             dict_partial[row[0]] = dict_partial[row[0]] + row[1:]
#
#     return dict_partial
#
# @task(returns=dict)
# def merge_AdjList(data1,data2):
#     print data1
#     print data2
#     for entry ,value in data2.iteritems():
#         if entry in data1:
#             data1[entry] += value
#         else:
#             data1[entry] = value
#     return data1
#
