#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.parameter     import *
from pycompss.api.task          import task
from pycompss.functions.reduce  import mergeReduce

import pandas as pd
import numpy  as np
import re
import itertools

class BagOfWords(object):
    """
        Bag-of-words (BoW):
        The bag-of-words model is a simplifying representation used in natural
        language processing and information retrieval. In this model, a text
        (such as a sentence or a document) is represented as the bag (multiset)
        of its words, disregarding grammar and even word order but keeping
        multiplicity.

        Methods:
            - fit():
            - transform():

    """
    def fit(self, train_set, params, numFrag):
        """
            Create a dictionary (vocabulary) of each word and its frequency in
            this set and in how many documents occured.

            :param train_set: A list of pandas dataframe with the documents
                              to be transformed;
            :param params:    A dictionary with some options:
                - attributes:   A list with columns which contains the tokenized
                                text/sentence;
                - minimum_df: Minimum number of how many documents a
                              word should appear;
                - minimum_tf: Minimum number of occurrences  of a word;
                - size:       Maximum size of the vocabulary.
                              If -1, no limits will be applied. (default, -1)
             :param numFrag:  A number of fragments;
            :return           Returns a model (dataframe) with the <word,tf,df>
        """

        if 'attributes' not in params:
            raise Exception("You must inform an `attributes` column.")

        params['minimum_df'] = params.get('minimum_df', 0)
        params['minimum_tf'] = params.get('minimum_tf', 0)
        params['size'] = params.get('size', -1)

        result_p = [ [] for f in range(numFrag) ]
        for f in range(numFrag):
            result_p[f]  =  wordCount(train_set[f], params)
        word_dic     = mergeReduce(merge_wordCount, result_p)
        vocabulary   = create_vocabulary(word_dic)

        if any([ params['minimum_df']>0,
                 params['minimum_tf']>0,
                 params['size']>0
                 ]):
            vocabulary  = filter_words(vocabulary, params)

        model = dict()
        model['algorithm'] = 'BagOfWords'
        model['model'] = vocabulary

        return  model


    def transform(self, test_set, model, params, numFrag):
        """
            transform():

            Perform the transformation of the data based in the model created.
            :param test_set:    A list of dataframes with the documents;
            :param model:       A model trained (grammar and its frequency);
            :param params:      A dictionary with the settings:
                - alias:        Name of the new column (default, BoW_vector);
                - attributes:   A list with columns which contains the tokenized
                                text/sentence;
            :param numFrag:     The number of fragments;
            :return   A list of pandas dataframe with the features transformed.
        """

        algorithm = model.get('algorithm','')
        if algorithm != 'BagOfWords':
            raise Exception("You must inform a valid BagOfWords model.")
        vocabulary = model['model']

        if 'attributes' not in params:
            raise Exception("You must inform an `attributes` column.")

        params['alias'] = params.get('alias', "BoW_vector")

        result_p = [ [] for f in range(numFrag) ]
        for f in range(numFrag):
            result_p[f] = transform_BoW(test_set[f], vocabulary, params)

        return result_p

@task(returns=dict)
def wordCount(data,params):
    partialResult = {}
    columns = params['attributes']
    #   first:   Number of all occorrences with term t
    #   second:  Number of diferent documents with term t
    #   third:   temporary - only to idetify the last occorrence
    i_line = 0
    for lines in data[columns].values:
        lines = np.array( list(itertools.chain(lines))).flatten()
        for token in lines:
            if token not in partialResult:
                partialResult[token] = [1, 1, i_line]
            else:
                partialResult[token][0] += 1
                if partialResult[token][2] != i_line:
                    partialResult[token][1] += 1
        i_line+=1
    return partialResult

@task(returns=dict)
def merge_wordCount(dic1, dic2):
    for k in dic2:
        if k in dic1:
            dic1[k][0] += dic2[k][0]
            dic1[k][1] += dic2[k][1]
        else:
            dic1[k] = dic2[k]
    return dic1


@task(returns=list)
def merge_lists(list1,list2):
    list1 = list1+list2
    return list1


@task(returns = list)
def create_vocabulary(word_dic):
    docs_list = [ [i[0], i[1][0], i[1][1] ] for i in word_dic.items()]
    names = ['Word','TotalFrequency','DistinctFrequency']
    voc = pd.DataFrame(docs_list, columns=names)
    return voc

@task(returns = list)
def filter_words(vocabulary, params):
    min_df = params['minimum_df']
    min_tf = params['minimum_tf']
    size = params['size']
    if min_df > 0:
       vocabulary = vocabulary.loc[vocabulary['DistinctFrequency']>=min_df]
    if min_tf > 0:
       vocabulary = vocabulary.loc[vocabulary['TotalFrequency']>=min_tf]
    if size > 0:
       vocabulary = \
            vocabulary.sort_values(['DistinctFrequency','TotalFrequency'],
                                    ascending=[1, 1]
                                    ).head(size)

    return vocabulary


@task(returns=list)
def transform_BoW( data, vocabulary, params):

    alias   = params['alias']
    columns = params['attributes']
    vector = np.zeros((len(data),len(vocabulary)),dtype=np.int)

    vocabulary = vocabulary['Word'].values
    data.reset_index(drop=True, inplace=True)
    for i, point in data.iterrows():
        lines = point[columns].values
        lines = np.array( list(itertools.chain(lines))).flatten()
        for e, w in enumerate(vocabulary):
            if w in lines:
                vector[i][e] = 1


    data[alias] = vector.tolist()

    return data
