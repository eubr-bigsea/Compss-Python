#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.text.tokenizer import TokenizerOperation
from functions.text.convert_words_to_vector import ConvertWordstoVectorOperation

import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.nan)
pd.options.display.max_colwidth = 500
pd.set_option('display.expand_frame_repr', False)


def main():
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
    data = TokenizerOperation().transform(data, settings, numFrag)

    settings = dict()
    settings['mode'] = 'BoW'
    settings['minimum_df'] = 3
    settings['minimum_tf'] = 4
    settings['size'] = -1
    settings['attributes'] = ['col_0']
    settings['alias'] = 'bow_0'
    settings, vocabulary = ConvertWordstoVectorOperation().preprocessing(data,
                                                               settings,
                                                               numFrag)
    print vocabulary
    data = ConvertWordstoVectorOperation().transform(data, vocabulary,
                                                     settings, numFrag)
    data = compss_wait_on(data)
    data = pd.concat(data, axis=0, sort=False)

    print data[['bow_0']]


main()
