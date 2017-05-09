#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks

import numpy as np

from text_functions import *


if __name__ == "__main__":

    test = np.asarray([['hello,how are you?I am and fine,thank you. And you?'],
            ['hi:okokok;fdf,fd,4'],
            ['hello,how are you?I am fine,thank you! And you?'],
            ['hi:okokok;fdf,fd,4']
            ])


    numFrag=4
    settings={}
    settings['min_token_length'] = 0
    settings['type'] = 'simple'
    t1 =  Tokenizer(test,settings,numFrag)


    stopwords = np.array(['am','parabola','4'])
    settings['news-stops-words'] =  np.array(["And","I"])
    settings['case-sensitive'] = True

    t = RemoveStopWords(t1,settings,stopwords,numFrag)


    params = {}
    params['minimum_tf'] = 3
    params['minimum_df'] = -1
    params['size']  = -1
    #t = get_vocabulary(t, params, numFrag)

    bow = Bag_of_Words()
    vocab = bow.fit(test,params,numFrag)
    t = bow.transform(test,vocab,numFrag)

    from pycompss.api.api import compss_wait_on

    t = compss_wait_on(t)
    print type(t)
    print t
    print "-----------"
    #for i in t:
    #    print i
