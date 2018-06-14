#!/usr/bin/python
# -*- coding: utf-8 -*-


import pandas as pd

import time

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.data.save_data import SaveOperation


pd.set_option('display.expand_frame_repr', False)


def main( ):
    start_time = time.time()
    # From HDFS
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['header'] = True
    settings['separator'] = ','

    data = ReadOperationHDFS().transform(filename, settings, numFrag)

    settings = dict()
    settings['filename'] = '/teste_save'
    settings['mode'] = 'overwrite'
    settings['header'] = True
    settings['format'] = 'csv'
    settings['storage'] = 'hdfs'
    data1 = SaveOperation().transform(data, settings, numFrag)
    data1 = compss_wait_on(data1)
    elapsed_time = time.time() - start_time

    for d in data1:
        print d

    print '{}s'.format(elapsed_time)


main()
