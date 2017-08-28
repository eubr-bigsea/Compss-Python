#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce



import pandas as pd
import numpy as np
import re
import itertools

#-------------------------------------------------------------------------------
# ConvertWordstoVector

def ConvertWordstoVectorOperation(data, params, numFrag):
    if params['mode'] == 'BoW':
        from BagOfWords import *
        BoW = BagOfWords()
        vocabulary  = BoW.fit(data,params,numFrag)
        data        = BoW.transform(data, vocabulary, params, numFrag)
        return data, vocabulary
    elif params['mode'] == 'TF-IDF':
        from TF_IDF import *
        tfidf = TF_IDF()
        vocabulary  = tfidf.fit(data,params,numFrag)
        data        = tfidf.transform(data, vocabulary, params, numFrag)
        return data, vocabulary

#-------------------------------------------------------------------------------
