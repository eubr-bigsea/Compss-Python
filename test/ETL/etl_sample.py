#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.sample import SampleOperation

if __name__ == '__main__':
    """Test sample function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/flights.csv'
    settings['separator'] = ','

    data0 = ReadOperationHDFS.transform(filename, settings, numFrag)

    # Testing Head option
    settings = dict()
    settings['type'] = 'head'
    # settings['per_value'] = 0.5
    settings['int_value'] = 50

    data1 = SampleOperation().transform(data0, settings, numFrag)
    data = compss_wait_on(data1)

    count = sum([len(d) for d in data])

    print count == 50

    # Testing Percent option
    # settings = dict()
    # settings['type'] = 'percent'
    # data1 = SampleOperation().transform(data0, settings, numFrag)

    # Testing Value option
    settings = dict()
    settings['type'] = 'value'
    # settings['per_value'] = 0.70
    settings['int_value'] = 300
    data1 = SampleOperation().transform(data0, settings, numFrag)

    data1 = compss_wait_on(data1)
    count = sum([len(d) for d in data1])

    print count == 300
