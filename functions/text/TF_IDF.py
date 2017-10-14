#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.parameter     import *
from pycompss.api.task          import task
from pycompss.functions.reduce  import mergeReduce

import pandas as pd
import numpy as np
import re
import itertools

class TF_IDF(object):

    """
        Term frequency-inverse document frequency (TF-IDF):
        It's a numerical statistic transformation that is intended to reflect
        how important a word is to a document in a collection or corpus.
    """

    def fit(self,data, params, numFrag):
        """
            Create a dictionary (vocabulary) of each word and its frequency in
            this set and in how many documents occured.

            :param train_set: A list of pandas dataframe with the
                              documents to be transformed.
            :param params:    A dictionary with some options:
                - attributes: A list with columns which contains the tokenized
                              text/sentence;
                - minimum_df: Minimum number of how many documents a
                              word should appear;
                - minimum_tf: Minimum number of occurrences  of a word;
                - size:       Maximum size of the vocabulary.
                              If -1, no limits will be applied. (default, -1)
            :return  A model (dataframe) with the <word,tf,df>
        """
        #Validation
        if 'attributes' not in params:
            raise Exception("You must inform an `attributes` column.")

        params['minimum_df'] = params.get('minimum_df', 0)
        params['minimum_tf'] = params.get('minimum_tf', 0)
        params['size'] = params.get('size', -1)

        result_p   = [self.wordCount(data[f], params) for f in range(numFrag)]
        word_dic   = mergeReduce(self.merge_wordCount, result_p)
        vocabulary = self.create_vocabulary(word_dic)

        if any([ params['minimum_df']>0,
                 params['minimum_tf']>0,
                 params['size']>0
                 ]):
            vocabulary  = self.filter_words(vocabulary,params)

        model = dict()
        model['algorithm'] = 'TF-IDF'
        model['model'] = vocabulary
        return  model



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
                    if partialResult[token][2]  != i_line:
                        partialResult[token][1] += 1
                        partialResult[token][2]  = i_line
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
        name_cols = ['Word','TotalFrequency','DistinctFrequency']
        vocabulary = pd.DataFrame(docs_list, columns=name_cols)
        return vocabulary

    @task(returns = list,isModifier = False)
    def filter_words(self, vocabulary, params):
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



    def transform(self, test_set, model, params, numFrag):
        """
            transform():

            Perform the transformation of the data based in the model created.
            :param test_set:    A list of dataframes with the documents
                                to transform;
            :param model:       A TF-IDF model (grammar and its frequency);
            :param params:      A dictionary with the settings:
                - alias:        Name of the new column (default, 'tfidf_vector');
                - attributes:   A list with columns which contains the tokenized
                                text/sentence;
            :param numFrag:     The number of fragments;
            :return   A list of pandas dataframe with the features transformed.
        """
        #Validation
        algorithm = model.get('algorithm','')
        if algorithm != 'TF-IDF':
            raise Exception("You must inform a valid BagOfWords model.")
        vocabulary = model['model']

        if 'attributes' not in params:
            raise Exception("You must inform an `attributes` column.")

        params['alias'] = params.get('alias', "tfidf_vector")

        counts = [self.count_records(test_set[f]) for f in range(numFrag)]
        count  = mergeReduce(self.mergeCount,counts)

        result = [ [] for f in range(numFrag) ]
        for f in range(numFrag):
            result[f] = \
                self.construct_TF_IDF(test_set[f], vocabulary, params, count)

        return result

    @task(returns = list,isModifier = False)
    def count_records(self, data):
        return len(data)

    @task(returns = list,isModifier = False)
    def mergeCount(self,data1,data2):
        return data1 + data2


    @task(returns = list,isModifier = False)
    def construct_TF_IDF(self, data, vocabulary, params, num_doc):
        """TF(t)  = (Number of times term t appears in a document)
                        / (Total number of terms in the document).
           IDF(t) = log( Total number of documents /
                        Number of documents with term t in it).
           Source: http://www.tfidf.com/
        """
        alias   = params['alias']
        columns = params['attributes']
        new_columns = np.zeros((len(data),len(vocabulary)),dtype=np.int)

        data[alias] = pd.Series(new_columns.tolist())

        vocab = vocabulary['Word'].values

        for i, point in data.iterrows():
            lines = point[columns].values
            lines = np.array( list(itertools.chain(lines))).flatten()

            for w in range(len(vocab)):
                token = vocab[w]
                if token in lines:
                    # TF = (Number of times term t appears in the document) /
                    #        (Total number of terms in the document).
                    nTimesTermT = np.count_nonzero(lines == token)
                    total = len(lines)
                    if total > 0:
                        tf =  float(nTimesTermT) / total
                    else:
                        tf = 0

                    # IDF = log_e(Total number of documents /
                    #            Number of documents with term t in it).
                    nDocsWithTermT = \
                            vocabulary.loc[ vocabulary['Word'] == token,
                                            'DistinctFrequency'
                                            ].item()
                    idf = np.log( float(num_doc) / nDocsWithTermT )
                    data.ix[i][alias][w] = tf*idf

        return data
