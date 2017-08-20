#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd
import sys

def StringIndexerOperation(data,settings,numFrag):
    inputCol  = settings['inputCol']
    outputCol = settings['outputCol']
    op = settings['IndexToString'] if 'IndexToString' in settings else False

    if not op:
        mapper = [getIndexs(data[f],inputCol) for f in range(numFrag)]
        mapper = mergeReduce(mergeMapper,mapper)

        data = [StringIndexer_p(data[f],inputCol,outputCol,mapper) for f in range(numFrag)]
        return [data, mapper]
    else:
        model = settings['model']
        data = [IndexToString_p(data[f],inputCol,outputCol, model) for f in range(numFrag)]
        return data


@task(returns=list)
def getIndexs(data,inputCol):
    x = data[inputCol].unique()
    x = filter(lambda v: v==v, x) #to remove nan values
    return x

@task(returns=list)
def mergeMapper(data1,data2):
    data1 = np.concatenate((data1, data2), axis=0)
    return np.unique(data1)

@task(returns=list)
def StringIndexer_p(data, inputCol, outputCol,mapper):
    news=[i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[outputCol] = data[inputCol].replace(to_replace=mapper, value=news,regex=False)
    return data

@task(returns=list)
def IndexToString_p(data,inputCol,outputCol,mapper):
    news=[i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[outputCol] = data[inputCol].replace(to_replace=news, value=mapper,regex=False)
    return data
