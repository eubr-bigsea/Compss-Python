#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"



#-------------------------------------------------------------------------------
# ConvertWordstoVector

def ConvertWordstoVectorOperation(data, params, numFrag):
    """
    ConvertWordstoVectorOperation():
    :param data:    A list of pandas dataframe with the documents
                    to be transformed.
    :param params:  A dictionary with some options:
        - mode:     'BoW' to use Bag-Of-Words, 'TF-IDF' to use Term frequency
                    inverse document frequency; (default, BoW)
        - all other specific parameters of each of the algorithms;
    :param numFrag: A number of fragments
    :return         The new dataframe with the transformed data and a model.
    """

    mode = params.get('mode','BoW')
    if mode not in ['BoW','TF-IDF']:
        raise Exception("You must inform a valid mode to convert "
                        "your text into vectors")

    if mode == 'BoW':
        from BagOfWords import BagOfWords
        BoW = BagOfWords()
        vocabulary  = BoW.fit(data,params,numFrag)
        data        = BoW.transform(data, vocabulary, params, numFrag)
        return data, vocabulary
    elif mode == 'TF-IDF':
        from TF_IDF import TF_IDF
        tfidf = TF_IDF()
        vocabulary  = tfidf.fit(data,params,numFrag)
        data        = tfidf.transform(data, vocabulary, params, numFrag)
        return data, vocabulary

#-------------------------------------------------------------------------------
