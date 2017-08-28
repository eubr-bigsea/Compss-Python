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

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
#-------------------------------------------------------------------------------

class BagOfWords(object):
    """
        Bag-of-words (BoW):
        The bag-of-words model is a simplifying representation used in natural
        language processing and information retrieval. In this model, a text
        (such as a sentence or a document) is represented as the bag (multiset)
        of its words, disregarding grammar and even word order but keeping
        multiplicity.

        Each row represents a document and each column correspond
        to the frequency of this word in the document.

    """
    def fit(self, train_set, params, numFrag):
        """
            Create a dictionary (vocabulary) of each word and its frequency in
            this set and in how many documents occured.

            :param input_txt: A np.array with the documents to transform.
            :param params:    A dictionary with some options:
                                - minimum_df
                                - minimum_tf
                                - vocabulary size
            :param numFrag: num fragments, if -1 data is considered chunked.
            :return  A merged numpy.array with the <vocabulary,tf,df>
        """

        result_p     = [self.wordCount(train_set[f], params) for f in range(numFrag)]
        word_dic     = mergeReduce(self.merge_wordCount, result_p)
        vocabulary   = self.create_vocabulary(word_dic)

        if params['minimum_df']>0 or params['minimum_tf']>0 or params['size']>0:
            vocabulary  = self.filter_words(vocabulary,params)


        return  vocabulary


    @task(returns=dict,isModifier = False)
    def wordCount(self,data,params):
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

    @task(returns=dict,isModifier = False)
    def merge_wordCount(self,dic1, dic2):
        for k in dic2:
            if k in dic1:
                dic1[k][0] += dic2[k][0]
                dic1[k][1] += dic2[k][1]
            else:
                dic1[k] = dic2[k]
        return dic1


    @task(returns=list,isModifier = False)
    def merge_lists(self,list1,list2):
        list1 = list1+list2
        return list1


    @task(returns = list,isModifier = False)
    def create_vocabulary(self,word_dic):
        docs_list = [ [i[0], i[1][0], i[1][1] ] for i in word_dic.items()]
        voc = pd.DataFrame(docs_list, columns=['Word','TotalFrequency','DistinctFrequency'])
        return voc

    @task(returns = list,isModifier = False)
    def filter_words(self,vocabulary, params):

        if params['minimum_df'] > 0:
            vocabulary = vocabulary.loc[vocabulary['DistinctFrequency'] >=  params['minimum_df']]
        if params['minimum_tf'] > 0:
            vocabulary = vocabulary.loc[vocabulary['TotalFrequency'] >=  params['minimum_tf']]
        if params['size'] > 0:
            vocabulary = vocabulary.sort_values(['DistinctFrequency','TotalFrequency'], ascending=[1, 1]).head(params['size'])

        return vocabulary

    def transform(self, test_set, vocabulary, params, numFrag):
        """
            :param train_set: A dataframe with the documents to transform.
            :param numFrag:   The number of fragments
            :param word_dic:  A model trained (grammar and its frequency).
            :param params: A dictionary with the settings:
                            - minimum_df
                            - minimum_tf
                            - vocabulary size
                            - vocabulary (if exists)
            :return A pandas dataframe with the features transformed.
        """

        partial_result = [self.transform_BoW(test_set[f], vocabulary, params) for f in range(numFrag)]

        return  partial_result

    @task(returns=list,isModifier = False)
    def transform_BoW(self, data, vocabulary, params):

        alias   = params['alias']
        columns = params['attributes']
        new_columns = np.zeros((len(data),len(vocabulary)),dtype=np.int)

        data[alias] = pd.Series(new_columns.tolist())

        vocabulary = vocabulary['Word'].values

        for i, point in data.iterrows():
            lines = point[columns].values
            lines = np.array( list(itertools.chain(lines))).flatten()
            for w in range(len(vocabulary)):
                if vocabulary[w] in lines:
                    data.ix[i][alias][w] = 1

        return data
