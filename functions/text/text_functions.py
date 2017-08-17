#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks
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
def RemoveStopWords(data,settings,stopwords,numFrag):
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

    result = [RemoveStopWords_part(data[i],settings,stopwords) for i in range(numFrag)]
    return result

@task(returns=list)
def RemoveStopWords_part( data, settings, stopwords):
    new_data = []
    columns = settings['attribute']
    alias   = settings['alias']
    #stopwords must be in 1-D
    stopwords  = np.reshape(pd.concat(stopwords).values , -1, order='C')
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


#-------------------------------------------------------------------------------


def Tokenizer(data,settings,numFrag):
    """
        Tokenization is the process of taking text (such as a sentence) and
        breaking it into individual terms (usually words). A simple Tokenizer
        class provides this functionality.

        :param data: A list
        :param settings: A dictionary with some configs:
                         - type: Type of the Tokenization (simple or regex). But
                         in moment only "simple" is accepted.
                         - min_token_length: minimun lenght of the token

        :return A new list
    """
    columns = settings['attributes']
    alias   = settings['alias']

    if settings['type'] == "simple":
        min_token_length = int(settings['min_token_length'])
        result = [Tokenizer_part(data[i],min_token_length,columns,alias) for i in range(numFrag)]

        return result

@task(returns=list)
def Tokenizer_part(data,min_token_length,columns,alias):

    result = []
    for line in data[columns].values:
        row = []
        for column in line:
            toks = re.split('[?!:;\s]|(?<!\d)[,.]|[,.](?!\d)', column)
            col = []
            for t in toks:
                if len(t)>min_token_length:
                    col.append(t)
            row.append(col)
        result.append(row)

    tmp = pd.DataFrame(result, columns=alias)

    result = pd.concat([data.reset_index(drop=True), tmp], axis=1)

    return result


#-------------------------------------------------------------------------------
#
@task(returns=dict)
def wordCount(data,params):
    partialResult = {}
    columns = params['attributes']
    i_line = 0
    for lines in data[columns].values:
        for col in lines:
            for token in col:
                    token = token.lower()
                    if token not in partialResult:
                        partialResult[token] = [1, 1, i_line]
                    else:
                        partialResult[token][0] += 1
                        if partialResult[token][2] != i_line:
                            partialResult[token][1] += 1
        i_line+=1
    return partialResult

@task(returns=dict)
def merge_wordCount(dic1,dic2):
    for k in dic2:
        if k in dic1:
            dic1[k][0] += dic2[k][0]
            dic1[k][1] += dic2[k][1]
        else:
            dic1[k] = dic2[k]
    return dic1


@task(returns=list)
def filter_words(dicts,params):
    new_dict = []

    if params['minimum_df'] > 0:
        new_dict = [item for item in dicts if item[1]>=params['minimum_df']]
    if params['minimum_tf'] > 0:
        new_dict = [item for item in dicts if item[2]>=params['minimum_tf']]
    #if params['size']> 0:  TO DO
    #    new_dict = [item for item in new_dict if item[0] not in stopwords_set]
    return np.asarray(new_dict)

@task(returns=list)
def merge_lists(list1,list2):
    list1 = list1+list2
    return list1

#-------------------------------------------------------------------------------
class Bag_of_Words(object):
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
    def fit(self,input_txt, params, numFrag):
        """
            get_vocabulary:

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

        result_p     = [ wordCount(input_txt[f],params) for f in range(numFrag)]
        word_dic     = mergeReduce(merge_wordCount, result_p)
        word_dic     = self.create_worddict(word_dic)

        if params['minimum_df']>0 or params['minimum_tf']>0 or params['size']>0:
            word_dic  = filter_words(word_dic,params)

        vocabulary     =  self.create_vocabulary(word_dic)
        return word_dic, vocabulary


    def transform(self,train_set, vocabulary,params, numFrag):
        """
            :param train_set: A np.array with the documents to transform.
            :param numFrag: num of fragments
            :param word_dic: A model trained (grammar and its frequency).
            :param params: A dictionary with the settings:
                            - minimum_df
                            - minimum_tf
                            - vocabulary size
                            - vocabulary (if exists)
            :return A np.array with the features transformed.
        """

        partial_result = [self.transform_BoW(train_set[f], vocabulary, params['attributes']) for f in range(numFrag)]
        #partial_result = compss_wait_on(partial_result)

        return  partial_result


    @task(returns=list,isModifier = False)
    def create_worddict(self,word_dic):
        return np.asarray([[i[0],i[1][0],i[1][1]] for i in word_dic.items()])

    @task(returns = list,isModifier = False)
    def create_vocabulary(self,word_dic):
        #print pd.DataFrame(data=word_dic,columns=['vocabulary','count_doc','count_all'])
        return pd.DataFrame(data=word_dic[:,0],columns=['vocabulary'])# np.asarray(word_dic)[:,0]


    #Duvida conceitual: Cada attributo sera considerado individualmente ou farà parte de algo maior


    @task(returns=list,isModifier = False)
    def transform_BoW(self, data, word_list,columns):
        table = []

        for row in data[columns].values:
            new_columns = []
            for col in row:
                for w in col:
            #     if word_list[f][0] in tokens:
            #         row.append(tokens.count(word_list[f][0]))
            #     else:
            #         row.append(0)
            # table.append(row)

        return np.asarray(table)
