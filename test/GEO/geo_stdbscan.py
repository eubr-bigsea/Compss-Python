#!/usr/bin/python
# -*- coding: utf-8 -*-

from functions.data.read_data import ReadOperationHDFS
from functions.geo.stdbscan.stdbscan import STDBSCAN
from functions.data.attributes_changer import AttributesChangerOperation
from pycompss.api.api import compss_wait_on
import pandas as pd


def main():
    # From HDFS

    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/taxi_geo100.csv'
    settings['header'] = True
    settings['separator'] = ';'

    data = ReadOperationHDFS().transform(filename, settings, numFrag)

    settings = dict()
    settings['attributes'] = ['DATATIME']
    settings['new_data_type'] = 'Date/time'
    data = AttributesChangerOperation().transform(data, settings, numFrag)

    settings = dict()
    settings['spatial_threshold'] = 1000  # meters
    settings['temporal_threshold'] = 60  # minutes
    settings['minPts'] = 2

    # columns
    settings['lat_col'] = 'LATITUDE'
    settings['lon_col'] = 'LONGITUDE'
    settings['datetime'] = 'DATATIME'
    settings['predCol'] = 'cluster'

    stdbscan = STDBSCAN()
    data = stdbscan.fit_predict(data, settings, numFrag)

    data = compss_wait_on(data)
    data = pd.concat(data, sort=False, axis=0)
    print data.head(100)
    print len(data)


main()
