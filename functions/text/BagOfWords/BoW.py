#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter    import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data   import chunks

from dist.text_functions import *

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

@task(returns=list)
def construct_BoW(data, word_list):

    table = []
    for entry in data:
        row = []
        tokens = entry.lower().split(" ")
        for f in range(len(word_list)):
            if word_list[f][0] in tokens:
                row.append(tokens.count(word_list[f][0]))
            else:
                row.append(0)
        table.append(row)

    return table

def transform_BoW(input_txt, params, numFrag, word_dic):
    """
        Bag-of-words (BoW):
        The bag-of-words model is a simplifying representation used in natural
        language processing and information retrieval. In this model, a text
        (such as a sentence or a document) is represented as the bag (multiset)
        of its words, disregarding grammar and even word order but keeping
        multiplicity.

        Each row represents a document and each column correspond
        to the frequency of this word in the document.

        :param input_txt: A np.array with the documents to transform.
        :param params:    A dictionary with some options:
                            'remove_accent'= True to remove the accents;
                            'remove_symbols' = True to remove the symbols;
        :param numFrag: num fragments, if -1 data is considered chunked.
        :param word_dic: A model trained (the grammar and the frequency).
        :return A np.array with the features transformed.
    """

    if params['remove_accent']  or params['remove_symbols']:
        params['clean_text'] = True

    from pycompss.api.api import compss_wait_on
    input_txt    = [ d for d in chunks(input_txt, len(input_txt)/numFrag)]

    if params['clean_text'] :
        input_txt  = [ clean_text(input_txt[f],params) for f in range(numFrag)]
    result_p = [ construct_BoW(input_txt[f], word_dic) for f in range(numFrag)]
    BoW       = [ mergeReduce(merge_lists, result_p) ]
    BoW = compss_wait_on(BoW)

    return BoW[0]



if __name__ == "__main__":
    numFrag = 4

    params  = {}
    params['min_occurrence'] = 4
    params['max_occurrence'] = 100

    params['remove_accent']  = True
    params['remove_symbols'] = True



    train_set = read_text("input_pos.txt",'\n',[])
    stopwords_set = read_text("stopwords.txt",'\n',[])

    word_dic = get_vocabulary(train_set,params,numFrag,stopwords_set)
    print word_dic
    print "Size of vocabulary %d" % len(word_dic)

    BoW = transform_BoW(train_set,params,numFrag,word_dic)

    thefile = open('output_BoW.txt', 'w')
    for item in BoW:
        thefile.write("%s\n" % item)
