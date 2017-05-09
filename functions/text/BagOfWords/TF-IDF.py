#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter    import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data   import chunks

from dist.text_functions import *
from BoW import *

import math
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

@task(returns=list)
def construct_TF_IDF(data, word_dic,num_doc):
    """TF(t)  = (Number of times term t appears in a document) / (Total number of terms in the document).
       IDF(t) = log(Total number of documents / Number of documents with term t in it).
       Source: http://www.tfidf.com/
    """

    vocabulary = [i[0] for i in word_dic] 
    idf = [math.log( float(num_doc) / ( i[2])) for i in word_dic]
    table = []

    for entry in data:
        row = []
        tokens = entry.lower().split(" ")
        for f in range(len(vocabulary)):
            if vocabulary[f] in tokens:
                tf = float(tokens.count(vocabulary[f]))/len(tokens)
                row.append(tf*idf[f])
            else:
                row.append(0)
        table.append(row)

    return table

def fit_TF_IDF(input_txt, params, numFrag, word_dic,num_doc):
    """
        Term frequency-inverse document frequency (TF-IDF):
        It's a numerical statistic transformation that is intended to reflect
        how important a word is to a document in a collection or corpus.


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
    TF_IDF_p    = [  construct_TF_IDF(input_txt[f], word_dic,num_doc) for f in range(numFrag)]
    TF_IDF  = [ mergeReduce(merge_lists, TF_IDF_p) ]
    TF_IDF = compss_wait_on(TF_IDF)

    return TF_IDF[0]



if __name__ == "__main__":
    numFrag = 4

    params  = {}
    params['min_occurrence'] = 4
    params['max_occurrence'] = 100

    params['remove_accent']  = True
    params['remove_symbols'] = True


    train_set = read_text("input_pos.txt",'\n',[])
    num_doc = len(train_set)
    print num_doc
    stopwords_set = read_text("stopwords.txt",'\n',[])

    word_dic = get_vocabulary(train_set,params,numFrag,stopwords_set)
    print word_dic
    print "Size of vocabulary %d" % len(word_dic)

    TF_IDF = fit_TF_IDF(train_set,params,numFrag,word_dic,num_doc)

    thefile = open('output_TF-IDF.txt', 'w')
    for item in TF_IDF:
        thefile.write("%s\n" % item)
