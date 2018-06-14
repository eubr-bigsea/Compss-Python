#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.text.tokenizer import TokenizerOperation

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
    for i in range(numFrag):
        data[i]['col_1'] = data[i]['col_0']
        print data[i]

    settings = dict()
    settings['min_token_length'] = 3
    settings['attributes'] = ['col_0', 'col_1']
    settings['alias'] = ['col_0_tok', 'col_1_tok']
    settings['type'] = 'regex'
    settings['expression'] = '[?!:;\s]'
    data = TokenizerOperation().transform(data, settings, numFrag)
    data = compss_wait_on(data)

    for d in data:
        print d[['col_0_tok', 'col_1_tok']]
    print data[0].columns


main()
