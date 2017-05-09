#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter    import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data   import chunks

import string
import re
import unicodedata
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# --------- WordCount
def WordCount(data):
    """
        WordCount:

        Do a Word Count with the follow schema: word, frequency in all the
        documents, number of documents with was occured.

        :param data: A np.array already splitted.
        :return A np.array with (word,total_frequency,number of documents)
    """
    from pycompss.api.api import compss_wait_on
    data    = [ d for d in chunks(data, len(data)/numFrag)]

    result_p     = [ wordCount(data[f]) for f in range(numFrag)]
    word_list    = [ mergeReduce(merge_wordCount, result_p) ]
    word_list = compss_wait_on(word_list)
    word_list = [[i[0],i[1][0],i[1][1]] for i in word_list[0].items()]
    return word_list


@task(returns=dict)
def wordCount(data):
    partialResult = {}

    i_line = 0
    for entry in data:
        tokens = entry.split(" ")
        tokens = [tok for tok in tokens if tok != ""]
        for token in tokens:
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

#----------------------------

def filter_accents(s):
    return ''.join((c for c in unicodedata.normalize('NFD', s.decode('UTF-8')) if unicodedata.category(c) != 'Mn'))

def filter_punct (ent):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    ent = regex.sub('', ent)
    return ent

@task(returns=list)
def filter_words(dicts,params,stopwords_set):
    new_dict = []
    if params['min_occurrence'] > 0:
        min_occurrence = params['min_occurrence']
        new_dict = [item for item in dicts if item[1]>=min_occurrence]
    if params['max_occurrence'] > 0:
        max_occurrence = params['max_occurrence']
        new_dict = [item for item in new_dict if item[1]<=max_occurrence]
    if len(stopwords_set)> 0:
        new_dict = [item for item in new_dict if item[0] not in stopwords_set]
    return new_dict

@task(returns=list)
def merge_lists(list1,list2):
    list1 = list1+list2
    return list1

@task(returns=list)
def clean_text(data,params):
    new_data = []
    for entry in data:
        if params['remove_accent']:
            entry = filter_accents(entry)
        if params['remove_symbols']:
            entry = filter_punct(entry)
        new_data.append(entry)

    return new_data
# ---------------------------------------------------------

def get_vocabulary(input_txt, params, numFrag, stopwords_set):
    """
        get_vocabulary:

        Create a dictionary (vocabulary) of each word and its frequency in
        this set and in how many documents occured.

        :param input_txt: A np.array with the documents to transform.
        :param params:    A dictionary with some options:
                            'remove_accent'= True to remove the accents;
                            'remove_symbols' = True to remove the symbols;
                            'min_occurrence' = number;
                            'max_occurrence' = integer;
        :param numFrag: num fragments, if -1 data is considered chunked.
        :param stopwords_set: Words to exclude in this vocabulary.
        :return A dictionary
    """


    if params['remove_accent']  or params['remove_symbols']:
        params['clean_text'] = True

    from pycompss.api.api import compss_wait_on
    input_txt    = [ d for d in chunks(input_txt, len(input_txt)/numFrag)]

    if params['clean_text']:
        input_txt    = [ clean_text(input_txt[f],params) for f in range(numFrag)]
    result_p     = [ wordCount(input_txt[f],params) for f in range(numFrag)]
    word_dic     = [ mergeReduce(merge_wordCount, result_p) ]
    word_dic = compss_wait_on(word_dic)
    word_dic = [[i[0],i[1][0],i[1][1]] for i in word_dic[0].items()]

    if params['min_occurrence'] > 0 or params['max_occurrence'] > 0 or len(stopwords_set)>0:
        dicts     = [ d for d in chunks(word_dic, len(word_dic)/numFrag)]
        result_p  = [filter_words(dicts[f],params,stopwords_set) for f in range(numFrag)]
        word_dic  = [ mergeReduce(merge_lists, result_p) ]

    word_dic=compss_wait_on(word_dic)

    return word_dic[0]




if __name__ == "__main__":
    numFrag = 4
    train_set = read_text("input_pos.txt",'\n',[])

    print WordCount(train_set)
