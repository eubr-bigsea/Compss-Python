#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.text.tokenizer import TokenizerOperation
from functions.text.remove_stopwords import RemoveStopWordsOperation

import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.nan)
pd.options.display.max_colwidth = 500
pd.set_option('display.expand_frame_repr', False)


def main( ):
    # From HDFS
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/text_data.txt'
    settings['header'] = False
    settings['separator'] = '\n'

    data = ReadOperationHDFS().transform(filename, settings, numFrag)

    data = compss_wait_on(data)
    data[2] = data[2][0:0]

    settings = dict()
    settings['min_token_length'] = 3
    settings['attributes'] = ['col_0']
    settings['alias'] = 'col_0'
    data1 = TokenizerOperation().transform(data, settings, numFrag)
    data1 = compss_wait_on(data1)
    for d in data1:
        print d.to_string()

    stopwords = [[] for _ in range(numFrag)]
    stopwords[0] = pd.DataFrame(['funny', 'clever', 'short'], columns=['TOK'])
    stopwords[1] = pd.DataFrame(['though', 'agreeably'], columns=['TOK'])
    stopwords[2] = pd.DataFrame(['somewhere'], columns=['TOK'])
    stopwords[3] = pd.DataFrame(['Gorgeously', 'segal'], columns=['TOK'])

    for s in stopwords:
        print s

    settings2 = dict()
    settings2['attributes'] = ['col_0']
    settings2['news-stops-words'] = ['Rock', '21st', 'minutes']
    settings2['case-sensitive'] = False
    settings2['col_words'] = 'TOK'

    import time
    start = time.time()
    data2 = RemoveStopWordsOperation().transform(data1, stopwords,
                                                 settings2, numFrag)
    data2 = compss_wait_on(data2)
    end = time.time()
    for d in data2:
        print d
    print "{}s".format(end-start)


main()

