#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

from pycompss.api.api import compss_wait_on

import pandas as pd
import numpy as np
import re


def filter_accents(s):
    return ''.join((c for c in unicodedata.normalize(\
                'NFD', s.decode('UTF-8')) if unicodedata.category(c) != 'Mn'))

def filter_punct (ent):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    ent = regex.sub('', ent)
    return ent


#-------------------------------------------------------------------------------
#
def RemoveStopWordsOperation(data,settings,stopwords,numFrag):
    """
        RemoveStopWords:
        Stop words are words which should be excluded from the input,
        typically because the words appear frequently and don’t carry
        as much meaning.

        :param data: A input (pandas's dataframe)
        :param settings: A dictionary with:
                            - ['news-stops-words']: A list with some stopwords
                                                        (wrote in Citron)
                            - ['case-sensitive']:   True or False
        :param stopwords: A list with some stopwords (as input)
        :return A new dataset
    """

    result = [ RemoveStopWords_part(data[i],settings,stopwords) for i in range(numFrag)]
    return result

@task(returns=list)
def RemoveStopWords_part( data, settings, stopwords):
    new_data = []
    columns = settings['attribute']
    alias   = settings['alias']
    #stopwords must be in 1-D
    stopwords  = np.reshape(np.array(stopwords), -1, order='C')
    new_stops  = np.reshape(settings['news-stops-words'], -1, order='C')
    stopwords  = np.concatenate((stopwords,new_stops), axis=0)

    if settings['case-sensitive']:
        for index, row in data.iterrows():
            col =[]
            for entry in row[columns]:
                col.append([tok for tok in tmp if tok not in stopwords])
            new_data.append(col)

    else:
        stopwords = [tok.lower() for tok in stopwords]
        print data
        for index, row in data.iterrows():
            col =[]
            for entry in row[columns]:
                tmp = [tok.lower() for tok in entry]
                col.append([tok for tok in tmp if tok not in stopwords])
            new_data.append(col)

    tmp = pd.DataFrame(new_data, columns=alias)
    result = pd.concat([data.reset_index(drop=True), tmp], axis=1)

    return result
