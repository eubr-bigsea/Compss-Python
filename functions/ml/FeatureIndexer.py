#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.parameter    import *
from pycompss.api.task         import task
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd


def FeatureIndexerOperation(data,settings,numFrag):
    """
        FeatureIndexerOperation():

        Indexes a feature by encoding a string column as a column
        containing indexes.

        :param data:            A list with numFrag pandas's dataframe;
        :param settings:        A dictionary that contains:
            - 'inputCol':       Field to perform the operation,
            - 'outputCol':      Alias to the converted field
                                (default, add a suffix '_indexed');
            - 'IndexToString':  False to convert String into Indexes (integers),
                                True to convert back this Indexes into
                                the original Strings (if True, you need inform
                                the 'model' too);
            - 'model':          A model of mapping created by the
                                FeatureIndexerOperation;
        :param numFrag:         A number of fragments;
        :return                 Returns a new dataframe with the indexed field.
    """

    #Validation step and inital settings
    if 'inputCol' not in settings:
       raise Exception("You must inform the `inputCol` field.")
    inputCol  = settings['inputCol']
    outputCol = settings.get('outputCol', "{}_indexed")
    mode = settings.get('IndexToString', False)

    if not mode:
        mapper = [getIndexs(data[f],inputCol) for f in range(numFrag)]
        mapper = mergeReduce(mergeMapper,mapper)
        data   = [StringIndexer_p(data[f], inputCol, outputCol, mapper)
                    for f in range(numFrag)]

        model = dict()
        model['algorithm'] = 'FeatureIndexerOperation'
        model['model'] = mapper
        return [data, model]
    else:

        #Validation step
        if 'model' not in settings:
           raise Exception("You must inform the `model` setting.")
        model = settings['model']
        model_name = model.get('algorithm','')
        if model_name != 'FeatureIndexerOperation':
            raise Exception("You must inform the valid `model`.")

        #Operation step:
        mapper = model['model']
        data  = [IndexToString_p(data[f], inputCol, outputCol, mapper)
                    for f in range(numFrag)]
        return data


@task(returns=list)
def getIndexs(data, inputCol):
    x = data[inputCol].dropna().unique()
    return x

@task(returns=list)
def mergeMapper(data1,data2):
    data1 = np.concatenate((data1, data2), axis=0)
    return np.unique(data1)

@task(returns=list)
def StringIndexer_p(data, inputCol, outputCol, mapper):
    news = [i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[outputCol] = data[inputCol].replace(to_replace=mapper, value=news)
    return data

@task(returns=list)
def IndexToString_p(data,inputCol,outputCol,mapper):
    news = [i for i in range(len(mapper))]
    mapper = mapper.tolist()
    data[outputCol] = data[inputCol].replace(to_replace=news, value=mapper)
    return data
