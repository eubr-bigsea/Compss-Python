#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.clean_missing import CleanMissingOperation
import pandas as pd

if __name__ == '__main__':
    """Test Intersection function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    settings['separator'] = ','
    filename = '/flights.csv'

    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)

    settings = dict()
    settings['attributes'] = ["CarrierDelay"]
    settings['cleaning_mode'] = "MEAN"
    data1 = CleanMissingOperation().transform(data0, settings, numFrag)

    # settings = dict()
    # settings['attributes'] = ["CarrierDelay", "WeatherDelay",
    #                           "NASDelay", "SecurityDelay"]
    # settings['cleaning_mode'] = "MEDIAN"
    # data1 = CleanMissingOperation().transform(data0, settings, numFrag)

    settings = dict()
    settings['attributes'] = ["LateAircraftDelay"]
    settings['cleaning_mode'] = "REMOVE_COLUMN"
    data1 = CleanMissingOperation().transform(data1, settings, numFrag)

    # settings = dict()
    # settings['attributes'] = ["CarrierDelay", "WeatherDelay",
    #                           "NASDelay", "SecurityDelay"]
    # settings['cleaning_mode'] = "VALUE"
    # settings['value'] = -1
    # data1 = CleanMissingOperation().transform(data0, settings, numFrag)
    #
    # settings = dict()
    # settings['attributes'] = ["CarrierDelay", "WeatherDelay",
    #                           "NASDelay", "SecurityDelay"]
    # settings['cleaning_mode'] = "MODE"
    # data1 = CleanMissingOperation().transform(data0, settings, numFrag)

    data1 = compss_wait_on(data1)
    data = pd.concat(data1, axis=0)
    print "REMOVE_COLUMN:", "LateAircraftDelay" not in data.columns
