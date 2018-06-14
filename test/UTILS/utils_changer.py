#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.data.attributes_changer import AttributesChangerOperation


pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(threshold=np.nan)
pd.options.display.max_colwidth = 500


def main( ):
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['separator'] = ','
    filename = '/flights.csv'

    data = ReadOperationHDFS().transform(filename, settings, numFrag)

    settings = dict()
    settings['attributes'] = ['Year']
    settings['new_data_type'] = 'double'
    data = AttributesChangerOperation().transform(data, settings, numFrag)

    data = compss_wait_on(data)
    data = pd.concat(data, sort=False, axis=0)
    print data.head(100)


main()

