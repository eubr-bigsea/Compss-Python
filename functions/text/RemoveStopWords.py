#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.parameter     import *
from pycompss.api.task          import task
from pycompss.functions.reduce  import mergeReduce

import pandas as pd
import numpy  as np


def RemoveStopWordsOperation(data, stopwords, settings, numFrag):
    """
        RemoveStopWordsOperation():

        Stop words are words which should be excluded from the input,
        typically because the words appear frequently and don’t carry
        as much meaning.

        :param data:      A list of pandas dataframe;
        :param stopwords: A list of pandas dataframe with stopwords,
                          one token by row (empty to don't use it);
        :param settings:  A dictionary with:
            - news-stops-words: A list with some stopwords (default, [])
            - case-sensitive:   True or False (default, True)
            - attributes:   A list with columns which contains the tokenized
                            text/sentence;
            - alias:        Name of the new column (default, tokenized_rm);
            - col_words:    Attribute of second data source with stop words
                            ;

        :return           Returns a list of pandas dataframe
    """


    if 'attributes' not in settings:
        raise Exception("You must inform an `attributes` column.")

    settings['news-stops-words'] = settings.get('news-stops-words', [])
    settings['case-sensitive']   = settings.get('case-sensitive', True)
    settings['alias'] = settings.get('alias', "tokenized_rm")

    len_stopwords = len(stopwords)
    if len_stopwords  == 0:
        stopwords = [[]]
    else:
        if 'col_words' not in settings:
            raise Exception("You must inform an `col_words` column.")

    # for f in range(numFrag):
    #     for s in range(len_stopwords):
    #         data[f] = RemoveStopWords_part(data[f], settings, stopwords[s], s)

    #It assumes that stopwords's dataframe can fit in memmory
    for f in range(numFrag):
        stopwords[f] = ReadStopWords(stopwords[f],settings)
    stopwords = mergeReduce(mergeStopWords,stopwords)

    result = [[] for f in range(numFrag)]
    for f in range(numFrag):
        result[f] = RemoveStopWords_part(data[f], settings, stopwords)
    return result

@task(returns=list)
def ReadStopWords(data1,settings):
    if len(data1)>0:
        data1 = np.reshape(data1[settings['col_words']], -1, order='C')
    else:
        data1 = np.array([])
    return data1

@task(returns=list)
def mergeStopWords(data1,data2):
    data1 = np.concatenate((data1,data2), axis=0)
    return data1


@task(returns=list)
def RemoveStopWords_part(data, settings, stopwords):


    columns = settings['attributes']
    alias   = settings['alias']

    #stopwords must be in 1-D
    new_stops = np.reshape( settings['news-stops-words'], -1, order='C')
    if len(stopwords) !=0:
        # stopwords = np.reshape(stopwords[settings['col_words']], -1, order='C')
        stopwords = np.concatenate((stopwords,new_stops), axis=0)
    else:
        stopwords = new_stops


    new_data = []
    if data.shape[0] > 0:
        if settings['case-sensitive']:
            stopwords = set(stopwords)
            for index, row in data.iterrows():
                col = []
                for entry in row[columns]:
                    #col.append([tok for tok in entry if tok not in stopwords])
                    col.append(list(set(entry).difference(stopwords)))
                new_data.append(col)

        else:
            stopwords = [tok.lower() for tok in stopwords]
            stopwords = set(stopwords)

            for index, row in data.iterrows():
                col = []
                for entry in row[columns]:
                    entry = [tok.lower() for tok in entry]
                    col.append(list(set(entry).difference(stopwords)))
                    # col.append([tok for tok in entry
                    #                 if tok.lower() not in stopwords])
                new_data.append(col)

        data[alias] = np.reshape(new_data, -1, order='C')
    return data
